#include "render_graph.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_set>
#include <utility>
namespace {
Rectangle FlippedSourceRectangle(const Texture2D& texture) {
    return {0.0f, 0.0f, static_cast<float>(texture.width), -static_cast<float>(texture.height)};
}

Rectangle FullTargetRectangle(int width, int height) {
    return {0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)};
}

void ConfigureRenderTargetSampling(RenderTexture2D& target) {
    if (target.texture.id == 0) return;
    SetTextureWrap(target.texture, TEXTURE_WRAP_CLAMP);
}

// rlLoadTexture/rlLoadTextureCubemap do this mapping internally for the
// formats they support; there's no rlgl equivalent for a
// GL_TEXTURE_2D_ARRAY, so CreateTextureArrayFramebuffer needs its own
// copy for the raw glTexImage3D call. Covers exactly the formats this
// codebase's ToRLFormat() table maps to -- extend both together if you
// add a new one. Assumes desktop GL (matches the rest of this file's use
// of glFramebufferTextureLayer/glTexImage3D); the GLES3 equivalents
// differ for a couple of these.
bool PixelFormatToGL(int rlPixelFormat, unsigned int& internalFormat, unsigned int& format, unsigned int& type) {
    switch (rlPixelFormat) {
        case RL_PIXELFORMAT_UNCOMPRESSED_GRAYSCALE:
            internalFormat = GL_R8;         format = GL_RED;  type = GL_UNSIGNED_BYTE; return true;
        case RL_PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA:
            internalFormat = GL_RG8;        format = GL_RG;   type = GL_UNSIGNED_BYTE; return true;
        case RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8:
            internalFormat = GL_RGBA8;      format = GL_RGBA; type = GL_UNSIGNED_BYTE; return true;
        case RL_PIXELFORMAT_UNCOMPRESSED_R32:
            internalFormat = GL_R32F;       format = GL_RED;  type = GL_FLOAT;         return true;
        case RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32:
            internalFormat = GL_RGBA32F;    format = GL_RGBA; type = GL_FLOAT;         return true;
        default:
            return false;
    }
}

// CreateComputePassFromKernel is a one-shot setup call (see its header
// doc), so a failure here means a real setup bug -- duplicate kernel
// name, unresolved uniform location, missing registry entry, etc. --
// and silently limping on with a half-bound pass only turns that bug
// into a much harder-to-find rendering glitch three files away.
void RequireKernelBindingStep(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("CreateComputePassFromKernel: failed to " + what);
}

// Maps a GeometryPass's PrimitiveMode to the GL enum glDrawArrays wants.
// No rlgl-level PrimitiveMode concept exists to defer to here -- rlgl's
// own primitive constants (RL_POINTS/RL_TRIANGLES/RL_LINES) are meant for
// rlBegin/rlEnd immediate-mode batching, not a raw glDrawArrays call over
// an externally-bound SSBO, so this maps straight to the GL enums instead.
unsigned int ToGLPrimitive(PrimitiveMode mode) {
    switch (mode) {
        case PrimitiveMode::Triangles: return GL_TRIANGLES;
        case PrimitiveMode::Lines:     return GL_LINES;
        case PrimitiveMode::Points:
        default:                      return GL_POINTS;
    }
}

// Rebuilds a name->index cache from scratch. Used after erase() on the
// backing vector, since erasing shifts every trailing element down by one
// index -- cheaper to just recompute than to patch each shifted entry, and
// erase is rare compared to the Find*() calls the cache exists to speed up.
template <typename T>
void RebuildNameIndex(const std::vector<T>& items, std::unordered_map<std::string, std::size_t>& indexOut) {
    indexOut.clear();
    indexOut.reserve(items.size());
    for (std::size_t i = 0; i < items.size(); ++i) indexOut[items[i].name] = i;
}

// Replaces what used to be three copies of this (StoreUniform for
// ProcessPass, StoreComputeUniform for ComputePass, StoreGeometryUniform
// for GeometryPass) -- the only thing they differed on was where the GL
// program id comes from, which is now PassProgramId()'s job (overloaded
// in render_graph.h for ShaderBoundPass and ComputePass).
//
// This is the actual "uniform registry with caching" piece: every Set*
// call funnels through here, and if the incoming value equals what's
// already cached for that name (via UniformValueEquals -- the equality
// raylib's own types don't give us), it's a no-op: no location lookup, no
// rlEnableShader/rlSetUniform/rlDisableShader round trip, nothing marked
// dirty. Only a genuine change reaches the GPU.
template<typename PassT, typename T>
bool ApplyCachedUniform(PassT& pass, const std::string& name, T value, int rlUniformType, unsigned int& currentShaderId) {
    UniformValue newValue{value};

    auto it = std::find_if(pass.uniforms.begin(), pass.uniforms.end(),
                            [&](const Uniform& u) { return u.name == name; });

    Uniform* uniform = nullptr;
    if (it != pass.uniforms.end()) {
        if (UniformValueEquals(it->value, newValue)) {
            it->dirty = false;
            return true; // already this value on the GPU -- nothing to push
        }
        it->value = newValue;
        it->dirty = true; // set now, not just after the push below -- so a
                           // GPU push that fails past this point (programId
                           // gone) still leaves this entry correctly
                           // flagged as needing one, instead of looking
                           // clean with a value the GPU was never given.
        uniform = &(*it);
    } else {
        // rlGetLocationUniform is the rlgl-level call raylib's
        // GetShaderLocation itself just forwards to -- using it directly
        // here so uniform lookup goes through the same layer as
        // everything else in this file.
        unsigned int programId = PassProgramId(pass);
        if (programId == 0) return false;
        const int location = rlGetLocationUniform(programId, name.c_str());
        if (location < 0) return false;
        pass.uniforms.emplace_back(newValue, name, location);
        uniform = &pass.uniforms.back();
        uniform->dirty = true;
    }

    unsigned int programId = PassProgramId(pass);
    if (programId == 0) return false;

    rlEnableShader(programId);
    if constexpr (std::is_same_v<T, Color>) {
        Vector4 normalized = ColorNormalize(value);
        rlSetUniform(uniform->location, &normalized, rlUniformType, 1);
    } else {
        rlSetUniform(uniform->location, &value, rlUniformType, 1);
    }
    rlDisableShader();
    currentShaderId = 0;
    uniform->dirty = false;
    return true;
}
TextureBinding* FindTextureBinding(ProcessPass& pass, const std::string& name) {
    auto iterator = std::find_if(pass.textureBindings.begin(), pass.textureBindings.end(),
                                  [&](const TextureBinding& binding) { return binding.name == name; });
    return iterator == pass.textureBindings.end() ? nullptr : &(*iterator);
}
TextureBinding* FindTextureBinding(GeometryPass& pass, const std::string& name) {
    auto iterator = std::find_if(pass.textureBindings.begin(), pass.textureBindings.end(),
                                  [&](const TextureBinding& binding) { return binding.name == name; });
    return iterator == pass.textureBindings.end() ? nullptr : &(*iterator);
}
bool AllocateRenderTargetWithFormat(RenderTexture2D& target, int width, int height, int format, bool withDepth) {
    target.id = rlLoadFramebuffer();
    if (target.id == 0) return false;

    target.texture.id = rlLoadTexture(nullptr, width, height, format, 1);
    if (target.texture.id == 0) {
        rlUnloadFramebuffer(target.id);
        target.id = 0;
        return false;
    }
    target.texture.width = width;
    target.texture.height = height;
    target.texture.mipmaps = 1;
    target.texture.format = format;

    rlEnableFramebuffer(target.id);
    rlFramebufferAttach(target.id, target.texture.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0);

    if (withDepth) {
        target.depth.id = rlLoadTextureDepth(width, height, true);
        if (target.depth.id == 0) {
            rlUnloadTexture(target.texture.id);
            rlUnloadFramebuffer(target.id);
            target = {};
            rlDisableFramebuffer();
            return false;
        }
        rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_RENDERBUFFER, 0);
    } else {
        target.depth.id = 0;
    }

    if (!rlFramebufferComplete(target.id)) {
        rlUnloadTexture(target.texture.id);
        if (target.depth.id != 0) rlUnloadTexture(target.depth.id);
        rlUnloadFramebuffer(target.id);
        target = {};
        rlDisableFramebuffer();
        return false;
    }

    rlDisableFramebuffer();
    ConfigureRenderTargetSampling(target);
    return true;
}
bool AllocateRenderTarget(RenderTexture2D& target, int width, int height) {
    target = LoadRenderTexture(width, height);
    ConfigureRenderTargetSampling(target);
    return target.id != 0;
}

bool AllocateScaledRenderTarget(RenderTexture2D& target, int baseWidth, int baseHeight, float scale, TextureFilter filter) {
    const int width = std::max(1, static_cast<int>(baseWidth * scale));
    const int height = std::max(1, static_cast<int>(baseHeight * scale));
    if (!AllocateRenderTarget(target, width, height)) return false;
    SetTextureFilter(target.texture, filter);
    return true;
}

void ClearRenderTarget(RenderTexture2D& target, Color color) {
    if (target.id == 0) return;
    BeginTextureMode(target);
    ClearBackground(color);
    EndTextureMode();
}

// ---- rlgl uniform helpers ----
// Applies one UniformValue via rlSetUniform. Caller MUST have already
// called rlEnableShader() for the owning program -- rlSetUniform operates
// on "whatever program rlgl currently has bound", it doesn't take a shader
// id itself. That's the main behavioral difference from raylib's
// SetShaderValue(), which enables/disables the shader internally on every
// single call (convenient, but means N uniforms = N bind/unbind pairs).
void ApplyUniformRLGL(int location, const UniformValue& value) {
    if (location < 0) return;
    std::visit([&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, float>) {
            rlSetUniform(location, &v, RL_SHADER_UNIFORM_FLOAT, 1);
        } else if constexpr (std::is_same_v<T, int>) {
            rlSetUniform(location, &v, RL_SHADER_UNIFORM_INT, 1);
        } else if constexpr (std::is_same_v<T, Vector2>) {
            rlSetUniform(location, &v, RL_SHADER_UNIFORM_VEC2, 1);
        } else if constexpr (std::is_same_v<T, Vector3>) {
            rlSetUniform(location, &v, RL_SHADER_UNIFORM_VEC3, 1);
        } else if constexpr (std::is_same_v<T, Vector4>) {
            rlSetUniform(location, &v, RL_SHADER_UNIFORM_VEC4, 1);
        } else if constexpr (std::is_same_v<T, Color>) {
            Vector4 normalized = ColorNormalize(v);
            rlSetUniform(location, &normalized, RL_SHADER_UNIFORM_VEC4, 1);
        }
        // std::monostate: nothing stored yet, nothing to apply.
    }, value);
}

// Binds a texture to a sampler2D/samplerCube/sampler3D/sampler2DArray
// uniform at an explicit texture unit, picking the right glBindTexture
// target from the binding's isCubemap/isTextureArray/isTexture3D flags
// instead of always going through rlEnableTexture (GL_TEXTURE_2D only).
// DispatchComputePass and the Geometry branch in Apply() do this same
// dispatch inline; this is it factored out so ExecutePass's two call
// sites (custom-framebuffer-target and dedicated/ping-pong) can share it
// rather than being hardcoded to 2D, which is what silently broke
// sampler3D/samplerCube/sampler2DArray bindings on ProcessPass.
// slot 0 is left alone deliberately -- rlgl's own batch renderer uses
// texture unit 0 internally for whatever DrawTexturePro() is drawing, so
// any *additional* sampler your shader declares (a noise texture, a LUT,
// a texture pulled from another pass) needs to live on slot 1+ or you'll
// silently stomp the primary input texture mid-draw.
void BindTextureToSamplerTyped(const TextureBinding& binding, unsigned int textureId, int slot) {
    if (binding.location < 0) return;
    rlActiveTextureSlot(slot);
    if (binding.isCubemap) {
        glBindTexture(GL_TEXTURE_CUBE_MAP, textureId);
    } else if (binding.isTextureArray) {
        glBindTexture(GL_TEXTURE_2D_ARRAY, textureId);
    } else if (binding.isTexture3D) {
        glBindTexture(GL_TEXTURE_3D, textureId);
    } else {
        rlEnableTexture(textureId);
    }
    rlSetUniform(binding.location, &slot, RL_SHADER_UNIFORM_INT, 1);
}
}

RenderGraph::~RenderGraph() {
    if (IsWindowReady()) Unload();
}

RenderGraph::RenderGraph(RenderGraph&& other) noexcept {
    *this = std::move(other);
}

RenderGraph& RenderGraph::operator=(RenderGraph&& other) noexcept {
    if (this == &other) return *this;
    if (IsWindowReady()) Unload();

    width_ = other.width_;
    height_ = other.height_;
    sceneTexture_ = other.sceneTexture_;
    pingPongBuffers_[0] = other.pingPongBuffers_[0];
    pingPongBuffers_[1] = other.pingPongBuffers_[1];
    currentOutput_ = other.currentOutput_;
    previewAllPasses_ = other.previewAllPasses_;

    // Move all resource containers
    shaders_ = std::move(other.shaders_);
    passes_ = std::move(other.passes_);
    groups_ = std::move(other.groups_);
    groupTemplates_ = std::move(other.groupTemplates_);
    groupInstances_ = std::move(other.groupInstances_);
    renderOrder_ = std::move(other.renderOrder_);
    computePrograms_ = std::move(other.computePrograms_);
    computePasses_ = std::move(other.computePasses_);
    computeGroups_ = std::move(other.computeGroups_);
    buffers_ = std::move(other.buffers_);
    framebuffers_ = std::move(other.framebuffers_);
    geometryPasses_ = std::move(other.geometryPasses_);
    dependencies_ = std::move(other.dependencies_);
    automaticOrdering_ = other.automaticOrdering_;
    uniformBuffers_ = std::move(other.uniformBuffers_);
    renderTexturePool_ = std::move(other.renderTexturePool_);    
    // Re‑point shader definitions for post‑process passes
    for (ProcessPass& pass : passes_) {
        auto it = std::find_if(shaders_.begin(), shaders_.end(),
            [&](const ProcessShader& shader) { return shader.name == pass.shaderName; });
        pass.shaderDefinition = it == shaders_.end() ? nullptr : &(*it);
    }

    // Re‑point program definitions for compute passes
    for (ComputePass& pass : computePasses_) {
        auto it = std::find_if(computePrograms_.begin(), computePrograms_.end(),
            [&](const ComputeProgram& program) { return program.name == pass.programName; });
        pass.programDefinition = it == computePrograms_.end() ? nullptr : &(*it);
    }

    // Re‑point shader definitions for geometry passes
    for (GeometryPass& pass : geometryPasses_) {
        auto it = std::find_if(shaders_.begin(), shaders_.end(),
            [&](const ProcessShader& shader) { return shader.name == pass.shaderName; });
        pass.shaderDefinition = it == shaders_.end() ? nullptr : &(*it);
    }

    // Reset the moved‑from object to a default, empty state
    other.width_ = 0;
    other.height_ = 0;
    other.sceneTexture_ = {};
    other.pingPongBuffers_[0] = {};
    other.pingPongBuffers_[1] = {};
    other.currentOutput_ = {};
    other.previewAllPasses_ = false;

    other.shaders_.clear();
    other.passes_.clear();
    other.groups_.clear();
    other.groupTemplates_.clear();
    other.groupInstances_.clear();
    other.renderOrder_.clear();
    other.computePrograms_.clear();
    other.computePasses_.clear();
    other.computeGroups_.clear();
    other.buffers_.clear();
    other.framebuffers_.clear();
    other.geometryPasses_.clear();
    other.uniformBuffers_.clear();
    other.renderTexturePool_.clear();
    return *this;
}

bool RenderGraph::Initialize(int width, int height) {
    if (width <= 0 || height <= 0) return false;
    if (sceneTexture_.id != 0) Unload();

    RenderTexture2D scene, pingA, pingB;
    const bool ok = AllocateRenderTarget(scene, width, height) &&
                    AllocateRenderTarget(pingA, width, height) &&
                    AllocateRenderTarget(pingB, width, height);

    if (!ok) {
        if (scene.id != 0) UnloadRenderTexture(scene);
        if (pingA.id != 0) UnloadRenderTexture(pingA);
        if (pingB.id != 0) UnloadRenderTexture(pingB);
        return false;
    }

    width_ = width;
    height_ = height;
    sceneTexture_ = scene;
    pingPongBuffers_[0] = pingA;
    pingPongBuffers_[1] = pingB;
    currentOutput_ = sceneTexture_.texture;

    // One shared VAO for every attributeless GeometryPass -- see its
    // member doc comment. Created once here rather than lazily on first
    // use so a mid-frame SetGeometryAttributelessDraw call never has to
    // allocate a GL object.
    if (attributelessVAO_ == 0) attributelessVAO_ = rlLoadVertexArray();

    return true;
}

void RenderGraph::ReleasePassGpuResources(ProcessPass& pass) {
    pass.textureBindings.clear();
    pass.uniforms.clear();

    if (pass.output.id != 0) {
        ReleaseToPool(pass.output, pass.outputFormat, pass.outputWithDepth);
        pass.output = {};
    }
    if (pass.feedback.id != 0) {
        UnloadRenderTexture(pass.feedback); // feedback may not be pooled? or could pool similarly
        pass.feedback = {};
    }
    if (pass.history.id != 0) {
        UnloadRenderTexture(pass.history);
        pass.history = {};
    }
    pass.shaderDefinition = nullptr;
}

void RenderGraph::Unload() {
    // 1. Post‑process passes: release GPU resources and CPU vectors.
    for (ProcessPass& pass : passes_) ReleasePassGpuResources(pass);
    passes_.clear();

    // 2. Groups: unload their output render textures.
    for (ProcessGroup& group : groups_) {
        if (group.output.id != 0) UnloadRenderTexture(group.output);
        group.output = {};
    }
    groups_.clear();

    // 3. Geometry passes: clear uniforms and callbacks (no GPU objects owned directly).
    for (GeometryPass& pass : geometryPasses_) {
        pass.uniforms.clear();
        pass.matrixUniforms.clear();
        pass.drawCallback = nullptr;
        pass.drawItemsProvider = nullptr;
        pass.targetFramebufferName.clear();
        pass.shaderDefinition = nullptr;
    }
    geometryPasses_.clear();

    // The shared attributeless VAO is the one GPU object this class owns
    // outside individual passes/buffers -- release it here alongside
    // everything else Unload() tears down.
    if (attributelessVAO_ != 0) {
        rlUnloadVertexArray(attributelessVAO_);
        attributelessVAO_ = 0;
    }

    // 4. Standard shaders (vertex/fragment) loaded via raylib.
    for (ProcessShader& shader : shaders_)
        if (shader.shader.id != 0) UnloadShader(shader.shader);
    shaders_.clear();

    // 5. Compute programs and buffers – raw rlgl objects.
    for (ComputeProgram& program : computePrograms_)
        if (program.id != 0) rlUnloadShaderProgram(program.id);
    computePrograms_.clear();

    for (ShaderBuffer& buffer : buffers_)
        if (buffer.id != 0) rlUnloadShaderBuffer(buffer.id);
    buffers_.clear();

    // 6. Framebuffers: unload all color attachments, depth attachment (renderbuffer or texture), and the FBO itself.
    for (Framebuffer& fb : framebuffers_) {
        for (auto& att : fb.colorAttachments) {
            if (att.textureId != 0) rlUnloadTexture(att.textureId);
        }
        if (fb.depthRenderbufferId != 0) rlUnloadTexture(fb.depthRenderbufferId);
        if (fb.depthTextureId != 0) rlUnloadTexture(fb.depthTextureId);
        if (fb.id != 0) rlUnloadFramebuffer(fb.id);
    }
    framebuffers_.clear();
    for (auto& buffer : uniformBuffers_)
        if (buffer.id != 0) glDeleteBuffers(1, &buffer.id);
    for (auto& entry : renderTexturePool_) {
        if (entry.texture.id != 0) UnloadRenderTexture(entry.texture);
    }
    //profile clear
    if (!queryIds_.empty()) {
        for (auto& pair : queryIds_) {
            if (pair.second != 0) glDeleteQueries(1, &pair.second);
        }
        queryIds_.clear();
    }
    gpuTimes_.clear();
    renderTexturePool_.clear();
    uniformBuffers_.clear();
    // 7. Other containers.
    groupTemplates_.clear();
    groupInstances_.clear();
    renderOrder_.clear();
    computePasses_.clear();
    computeGroups_.clear();
    dependencies_.clear();
    computeWrittenTextures_.clear();
    barrierFlushedTextures_.clear();
    automaticOrdering_ = false;
    // 8. Ping‑pong buffers.
    for (RenderTexture2D& buffer : pingPongBuffers_) {
        if (buffer.id != 0) UnloadRenderTexture(buffer);
        buffer = {};
    }

    // 9. Scene texture(s).
    if (sceneTexture_.id != 0) UnloadRenderTexture(sceneTexture_);
    sceneTexture_ = {};
    for (auto& [name, tex] : sceneTextures_) {
        if (tex.id != 0) UnloadRenderTexture(tex);
    }
    sceneTextures_.clear();
    currentOutput_ = {};

    // 10. Reset state.
    lastError_.clear();
    currentShaderId_ = 0;
    width_ = 0;
    height_ = 0;
    previewAllPasses_ = false;
}
bool RenderGraph::AllocatePassOutput(ProcessPass& pass) {
    int targetWidth = std::max(1, static_cast<int>(width_ * pass.resolutionScale));
    int targetHeight = std::max(1, static_cast<int>(height_ * pass.resolutionScale));

    RenderTexture2D newTarget = AcquirePooledTexture(targetWidth, targetHeight,
                                                     pass.outputFilter,
                                                     pass.outputFormat,
                                                     pass.outputWithDepth);
    if (newTarget.id == 0) return false;

    if (pass.output.id != 0) {
        ReleaseToPool(pass.output, pass.outputFormat, pass.outputWithDepth);
    }
    pass.output = newTarget;
    return true;
}
bool RenderGraph::AllocateGroupOutput(ProcessGroup& group) {
    RenderTexture2D newTarget;

    if (!AllocateRenderTarget(newTarget, width_, height_)) return false;
    SetTextureFilter(newTarget.texture, TEXTURE_FILTER_BILINEAR);

    if (group.output.id != 0) UnloadRenderTexture(group.output);
    group.output = newTarget;
    return true;
}

void RenderGraph::SetError(const std::string& message) const {
    lastError_ = message;
}
// render_graph.cpp

// The exact same fragment shader raylib uses internally for default drawing.
// It samples 'texture0' and multiplies by 'colDiffuse' (which is WHITE by default).
static const char* DEFAULT_FRAGMENT_SHADER = R"(
#version 330
in vec2 fragTexCoord;
out vec4 fragColor;
uniform sampler2D texture0;
uniform vec4 colDiffuse;
void main() {
    fragColor = texture(texture0, fragTexCoord) * colDiffuse;
}
)";

ShaderHandle RenderGraph::CreateDefaultShader(const std::string& name) {
    if (width_ <= 0 || height_ <= 0 || name.empty()) {
        SetError("AddDefaultShader: invalid arguments or graph not initialized (name='" + name + "')");
        return ShaderHandle();
    }

    if (std::any_of(shaders_.begin(), shaders_.end(),
        [&](const ProcessShader& shader) { return shader.name == name; })) {
        SetError("AddDefaultShader: a shader named '" + name + "' already exists");
        return ShaderHandle();
    }

    ProcessShader shader;
    shader.name = name;
    shader.fragmentShaderPath = ""; // Not used for default
    shader.vertexShaderPath = "";   // Not used for default

    // LoadShaderFromMemory(vertexSrc, fragmentSrc)
    // Passing nullptr for vertexSrc tells raylib to use its built‑in default vertex shader.
    shader.shader = LoadShaderFromMemory(nullptr, DEFAULT_FRAGMENT_SHADER);

    if (shader.shader.id == 0) {
        SetError("AddDefaultShader: failed to compile default shader '" + name + "'");
        return ShaderHandle();
    }

    shaders_.push_back(std::move(shader));
    shadersIndex_[name] = shaders_.size() - 1;

    // Re‑resolve shader pointers for any existing passes (just like AddShader does).
    for (ProcessPass& pass : passes_) {
        auto it = std::find_if(shaders_.begin(), shaders_.end(),
            [&](const ProcessShader& candidate) { return candidate.name == pass.shaderName; });
        pass.shaderDefinition = it == shaders_.end() ? nullptr : &(*it);
    }

    return ShaderHandle(name);
}
ShaderHandle RenderGraph::CreateShader(const std::string& name,
                                    const std::string& fragmentShaderPath,
                                    const std::string& vertexShaderPath) {
    if (width_ <= 0 || height_ <= 0 || name.empty() || fragmentShaderPath.empty()) {
        SetError("AddShader: invalid arguments or graph not initialized (name='" + name + "')");
        return ShaderHandle();
    }
    if (std::any_of(shaders_.begin(), shaders_.end(),
        [&](const ProcessShader& shader) { return shader.name == name; })) {
        SetError("AddShader: a shader named '" + name + "' already exists");
        return ShaderHandle();
    }

    // Deliberately still raylib's LoadShader rather than raw
    // rlCompileShader + rlLoadShaderProgram. LoadShader does two things
    // beyond linking: it supplies raylib's built-in default vertex shader
    // source when vertexShaderPath is empty, and it populates
    // shader.locs[] for the attribute/uniform slots rlgl's own batch
    // renderer expects (SHADER_LOC_VERTEX_POSITION, SHADER_LOC_MATRIX_MVP,
    // SHADER_LOC_COLOR_DIFFUSE, etc.) -- without those locs, DrawTexturePro
    // silently breaks or draws black/garbage geometry when a custom shader
    // is active. Reimplementing that population correctly from raw rlgl
    // calls would mean guessing at raylib's default location-naming
    // convention with no way for me to verify it compiles/links correctly
    // on your end, so this one function stays at the raylib level; every
    // *use* of the resulting shader.id below goes through rlgl directly.
    ProcessShader shader;
    shader.name = name;
    shader.fragmentShaderPath = fragmentShaderPath;
    shader.vertexShaderPath = vertexShaderPath;
    shader.shader = LoadShader(vertexShaderPath.empty() ? nullptr : vertexShaderPath.c_str(),
                               fragmentShaderPath.c_str());
    if (shader.shader.id == 0) {
        SetError("AddShader: failed to compile/link shader '" + name + "' (frag='" +
                  fragmentShaderPath + "', vert='" + vertexShaderPath + "')");
        return ShaderHandle();
    }

    shaders_.push_back(std::move(shader));
    shadersIndex_[name] = shaders_.size() - 1;
    for (ProcessPass& pass : passes_) {
        auto it = std::find_if(shaders_.begin(), shaders_.end(),
            [&](const ProcessShader& candidate) { return candidate.name == pass.shaderName; });
        pass.shaderDefinition = it == shaders_.end() ? nullptr : &(*it);
    }
    return ShaderHandle(name);
}

bool RenderGraph::DestroyShader(const std::string& name) {
    auto it = std::find_if(shaders_.begin(), shaders_.end(),
        [&](const ProcessShader& shader) { return shader.name == name; });
    if (it == shaders_.end()) return false;

    // Block deletion if used by any post‑process pass
    for (const ProcessPass& pass : passes_)
        if (pass.shaderName == name) return false;

    // Block deletion if used by any geometry pass
    for (const GeometryPass& pass : geometryPasses_)
        if (pass.shaderName == name) return false;

    // Block deletion if used by any group template
    for (const ProcessGroupTemplate& groupTemplate : groupTemplates_)
        for (const ProcessPassTemplate& passTemplate : groupTemplate.passes)
            if (passTemplate.shaderName == name) return false;

    // Safe to Destroy
    if (it->shader.id != 0) UnloadShader(it->shader);
    shaders_.erase(it);
    RebuildNameIndex(shaders_, shadersIndex_);

    // erase() shifts every shader after the Destroyd one down by one slot,
    // which silently invalidates any pass's cached shaderDefinition pointer
    // into that tail. Re-resolve them the same way AddShader() does after
    // its own push_back, so no pass is left pointing at shifted data.
    for (ProcessPass& pass : passes_) {
        auto shaderIt = std::find_if(shaders_.begin(), shaders_.end(),
            [&](const ProcessShader& candidate) { return candidate.name == pass.shaderName; });
        pass.shaderDefinition = shaderIt == shaders_.end() ? nullptr : &(*shaderIt);
    }
    for (GeometryPass& geoPass : geometryPasses_) {
        auto shaderIt = std::find_if(shaders_.begin(), shaders_.end(),
            [&](const ProcessShader& candidate) { return candidate.name == geoPass.shaderName; });
        geoPass.shaderDefinition = shaderIt == shaders_.end() ? nullptr : &(*shaderIt);
    }
    return true;
}
PassHandle RenderGraph::CreateTextureCopyPass(const std::string& name,
                                           float resolutionScale,
                                           TextureFilter filter) {
    if (width_ <= 0 || height_ <= 0) {
        SetError("AddTextureCopyPass: graph not initialized");
        return PassHandle();
    }

    // Internal name for the default pass‑through shader.
    // Must be unique – choose something unlikely to conflict with user shaders.
    static const std::string internalShaderName = "__internal_texture_copy_shader__";
    static ShaderHandle defaultShaderHandle; // cached handle

    // Lazy creation: only compile the shader once.
    if (!defaultShaderHandle.IsValid()) {
        defaultShaderHandle = CreateDefaultShader(internalShaderName);
        if (!defaultShaderHandle.IsValid()) {
            SetError("AddTextureCopyPass: failed to create default shader");
            return PassHandle();
        }
    }

    // Add the actual process pass using that shader.
    PassHandle pass = CreatePass(name, defaultShaderHandle.name, true);
    if (!pass) {
        SetError("AddTextureCopyPass: failed to add pass instance '" + name + "'");
        return PassHandle();
    }

    // Apply resolution scaling and filtering.
    if (resolutionScale != 1.0f) {
        if (!SetResolutionScale(name, resolutionScale)) {
            DestroyPass(name);
            SetError("AddTextureCopyPass: failed to set resolution scale");
            return PassHandle();
        }
    }
    if (!SetOutputFilter(name, filter)) {
        DestroyPass(name);
        SetError("AddTextureCopyPass: failed to set output filter");
        return PassHandle();
    }

    return pass;
}
PassHandle RenderGraph::CreatePass(const std::string& name,
                                                const std::string& shaderName,
                                                bool persistentOutput) {
    if (width_ <= 0 || height_ <= 0 || name.empty() || shaderName.empty() ||
        name == SceneSourceName() || FindPass(name) || FindGroup(name))
        return PassHandle();

    auto shaderIt = std::find_if(shaders_.begin(), shaders_.end(),
        [&](const ProcessShader& shader) { return shader.name == shaderName; });
    if (shaderIt == shaders_.end() || shaderIt->shader.id == 0)
        return PassHandle();

    ProcessPass pass;
    pass.name = name;
    pass.shaderName = shaderName;
    pass.shaderDefinition = &(*shaderIt);
    pass.persistentOutput = persistentOutput;

    if ((persistentOutput || previewAllPasses_) && !AllocatePassOutput(pass))
        return PassHandle();

    passes_.push_back(std::move(pass));
    passesIndex_[name] = passes_.size() - 1;
    renderOrder_.push_back({RenderNodeType::Pass, name});
    return PassHandle(name);
}

PassHandle RenderGraph::CreatePassFromFile(const std::string& name,
                                        const std::string& fragmentShaderPath,
                                        const std::string& vertexShaderPath,
                                        bool persistentOutput) {
    if (!CreateShader(name, fragmentShaderPath, vertexShaderPath))
        return PassHandle();
    return CreatePass(name, name, persistentOutput);
}

bool RenderGraph::DestroyPass(const std::string& name) {
    auto iterator = std::find_if(passes_.begin(), passes_.end(),
                                 [&](const ProcessPass& pass) { return pass.name == name; });

    if (iterator == passes_.end()) return false;

    ReleasePassGpuResources(*iterator);
    passes_.erase(iterator);
    RebuildNameIndex(passes_, passesIndex_);

    for (ProcessGroup& group : groups_) {
        group.passNames.erase(
            std::remove(group.passNames.begin(), group.passNames.end(), name),
            group.passNames.end()
        );
    }

    renderOrder_.erase(
        std::remove_if(renderOrder_.begin(), renderOrder_.end(),
                       [&](const RenderNode& node) {
                           return node.type == RenderNodeType::Pass && node.name == name;
                       }),
        renderOrder_.end()
    );

    for (ProcessPass& pass : passes_)
        for (TextureBinding& binding : pass.textureBindings)
            if (binding.sourcePassName == name) binding.sourcePassName.clear();

    for (ProcessGroup& group : groups_)
        if (group.sourceName == name) group.sourceName.clear();

    currentOutput_ = sceneTexture_.texture;
    return true;
}
bool RenderGraph::AddPassClass(const std::string& passName, const std::string& className) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;
    if (std::find(pass->classNames.begin(), pass->classNames.end(), className) != pass->classNames.end())
        return false; // already has it
    pass->classNames.push_back(className);
    return true;
}

bool RenderGraph::RemovePassClass(const std::string& passName, const std::string& className) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;
    auto it = std::find(pass->classNames.begin(), pass->classNames.end(), className);
    if (it == pass->classNames.end()) return false;
    pass->classNames.erase(it);
    return true;
}

bool RenderGraph::AddComputeClass(const std::string& passName, const std::string& className) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    if (std::find(pass->classNames.begin(), pass->classNames.end(), className) != pass->classNames.end())
        return false;
    pass->classNames.push_back(className);
    return true;
}

bool RenderGraph::RemoveComputeClass(const std::string& passName, const std::string& className) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    auto it = std::find(pass->classNames.begin(), pass->classNames.end(), className);
    if (it == pass->classNames.end()) return false;
    pass->classNames.erase(it);
    return true;
}

bool RenderGraph::AddGeometryClass(const std::string& passName, const std::string& className) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;
    if (std::find(pass->classNames.begin(), pass->classNames.end(), className) != pass->classNames.end())
        return false;
    pass->classNames.push_back(className);
    return true;
}

bool RenderGraph::RemoveGeometryClass(const std::string& passName, const std::string& className) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;
    auto it = std::find(pass->classNames.begin(), pass->classNames.end(), className);
    if (it == pass->classNames.end()) return false;
    pass->classNames.erase(it);
    return true;
}
GroupHandle RenderGraph::CreateProcessGroup(const std::string& name) {
    if (width_ <= 0 || height_ <= 0 ||
        name.empty() ||
        name == SceneSourceName() ||
        FindPass(name) ||
        FindGroup(name)) {
        return GroupHandle();
    }

    ProcessGroup group;
    group.name = name;

    if (!AllocateGroupOutput(group))
        return GroupHandle();

    groups_.push_back(std::move(group));
    renderOrder_.push_back({RenderNodeType::Group, name});

    return GroupHandle(name);
}
bool RenderGraph::DestroyProcessGroup(const std::string& name) {
    auto iterator = std::find_if(groups_.begin(), groups_.end(),
                                 [&](const ProcessGroup& group) { return group.name == name; });

    if (iterator == groups_.end()) return false;

    if (iterator->output.id != 0) UnloadRenderTexture(iterator->output);

    groups_.erase(iterator);

    renderOrder_.erase(
        std::remove_if(renderOrder_.begin(), renderOrder_.end(),
                       [&](const RenderNode& node) {
                           return node.type == RenderNodeType::Group && node.name == name;
                       }),
        renderOrder_.end()
    );

    for (ProcessPass& pass : passes_)
        for (TextureBinding& binding : pass.textureBindings)
            if (binding.sourcePassName == name) binding.sourcePassName.clear();

    for (ProcessGroup& group : groups_)
        if (group.sourceName == name) group.sourceName.clear();

    return true;
}

void RenderGraph::Resize(int width, int height) {
    if (width <= 0 || height <= 0 ||
        (width == width_ && height == height_)) {
        return;
    }

    // Clear the render target pool because sizes are changing
    for (auto& entry : renderTexturePool_) {
        if (entry.texture.id != 0) UnloadRenderTexture(entry.texture);
    }
    renderTexturePool_.clear();

    RenderTexture2D newScene, newPingA, newPingB;

    if (!AllocateRenderTarget(newScene, width, height) ||
        !AllocateRenderTarget(newPingA, width, height) ||
        !AllocateRenderTarget(newPingB, width, height)) {

        if (newScene.id != 0) UnloadRenderTexture(newScene);
        if (newPingA.id != 0) UnloadRenderTexture(newPingA);
        if (newPingB.id != 0) UnloadRenderTexture(newPingB);
        return;
    }

    std::vector<RenderTexture2D> newPassOutputs(passes_.size());
    std::vector<RenderTexture2D> newPassFeedback(passes_.size());
    std::vector<RenderTexture2D> newPassHistory(passes_.size());
    std::vector<RenderTexture2D> newGroupOutputs(groups_.size());
    std::unordered_map<std::string, RenderTexture2D> newSceneTextures;

    auto releaseStaged = [&]() {
        for (RenderTexture2D& allocated : newPassOutputs)
            if (allocated.id != 0) UnloadRenderTexture(allocated);

        for (RenderTexture2D& allocated : newPassFeedback)
            if (allocated.id != 0) UnloadRenderTexture(allocated);

        for (RenderTexture2D& allocated : newPassHistory)
            if (allocated.id != 0) UnloadRenderTexture(allocated);

        for (RenderTexture2D& allocated : newGroupOutputs)
            if (allocated.id != 0) UnloadRenderTexture(allocated);

        for (auto& [name, tex] : newSceneTextures)
            if (tex.id != 0) UnloadRenderTexture(tex);

        UnloadRenderTexture(newScene);
        UnloadRenderTexture(newPingA);
        UnloadRenderTexture(newPingB);
    };

    // Reallocate named scene textures at the same new dimensions as the
    // default scene. (If per-source custom sizes are ever needed, this
    // would need to remember each source's own width/height instead.)
    for (const auto& [name, oldTex] : sceneTextures_) {
        RenderTexture2D newTex;
        if (oldTex.id != 0) {
            if (!AllocateRenderTarget(newTex, width, height)) {
                releaseStaged();
                return;
            }
        }
        newSceneTextures[name] = newTex;
    }

    // Reallocate pass outputs using their custom format/depth settings
    for (std::size_t i = 0; i < passes_.size(); ++i) {
        if (passes_[i].output.id != 0) {
            int targetWidth = std::max(1, static_cast<int>(width * passes_[i].resolutionScale));
            int targetHeight = std::max(1, static_cast<int>(height * passes_[i].resolutionScale));

            if (!AllocateRenderTargetWithFormat(newPassOutputs[i], targetWidth, targetHeight,
                                                passes_[i].outputFormat,
                                                passes_[i].outputWithDepth)) {
                releaseStaged();
                return;
            }
            SetTextureFilter(newPassOutputs[i].texture, passes_[i].outputFilter);
        }

        if (passes_[i].feedback.id != 0) {
            // Feedback targets remain using default format (or could be extended later)
            if (!AllocateScaledRenderTarget(
                newPassFeedback[i],
                width,
                height,
                passes_[i].resolutionScale,
                passes_[i].outputFilter)) {
                releaseStaged();
                return;
            }

            ClearRenderTarget(newPassFeedback[i], BLANK);
        }

        if (passes_[i].history.id != 0) {
            if (!AllocateScaledRenderTarget(
                newPassHistory[i],
                width,
                height,
                passes_[i].resolutionScale,
                passes_[i].outputFilter)) {
                releaseStaged();
                return;
            }

            ClearRenderTarget(newPassHistory[i], BLANK);
        }
    }

    // Reallocate group outputs (unchanged, default format)
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (groups_[i].output.id != 0 &&
            !AllocateRenderTarget(newGroupOutputs[i], width, height)) {
            releaseStaged();
            return;
        }

        if (newGroupOutputs[i].id != 0)
            SetTextureFilter(newGroupOutputs[i].texture, TEXTURE_FILTER_BILINEAR);
    }

    // Replace scene and ping-pong buffers
    UnloadRenderTexture(sceneTexture_);
    sceneTexture_ = newScene;

    UnloadRenderTexture(pingPongBuffers_[0]);
    pingPongBuffers_[0] = newPingA;

    UnloadRenderTexture(pingPongBuffers_[1]);
    pingPongBuffers_[1] = newPingB;

    // Replace named scene textures
    for (auto& [name, tex] : sceneTextures_) {
        if (tex.id != 0) UnloadRenderTexture(tex);
    }
    sceneTextures_ = std::move(newSceneTextures);

    // Replace pass outputs, feedbacks, and history buffers
    for (std::size_t i = 0; i < passes_.size(); ++i) {
        if (newPassOutputs[i].id != 0) {
            UnloadRenderTexture(passes_[i].output);
            passes_[i].output = newPassOutputs[i];
        }

        if (newPassFeedback[i].id != 0) {
            UnloadRenderTexture(passes_[i].feedback);
            passes_[i].feedback = newPassFeedback[i];
        }

        if (newPassHistory[i].id != 0) {
            UnloadRenderTexture(passes_[i].history);
            passes_[i].history = newPassHistory[i];
        }
    }

    // Replace group outputs
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (newGroupOutputs[i].id != 0) {
            UnloadRenderTexture(groups_[i].output);
            groups_[i].output = newGroupOutputs[i];
        }
    }

    width_ = width;
    height_ = height;
    currentOutput_ = sceneTexture_.texture;
}
void RenderGraph::BeginScene(Color clearColor) {
    if (sceneTexture_.id == 0) return;
    BeginTextureMode(sceneTexture_);
    ClearBackground(clearColor);
}

void RenderGraph::BeginScene(const std::string& sceneName, Color clearColor) {
    if (sceneName.empty() || sceneName == SceneSourceName()) {
        BeginScene(clearColor);
        return;
    }
    auto it = sceneTextures_.find(sceneName);
    if (it == sceneTextures_.end() || it->second.id == 0) {
        SetError("BeginScene: no named scene texture '" + sceneName + "' (call AddSceneTexture first)");
        return;
    }
    BeginTextureMode(it->second);
    ClearBackground(clearColor);
}

void RenderGraph::EndScene() {
    EndTextureMode();
}

bool RenderGraph::CreateSceneTexture(const std::string& sourceName, int width, int height) {
    if (sourceName.empty() || sourceName == SceneSourceName() || width <= 0 || height <= 0) {
        SetError("AddSceneTexture: invalid name or dimensions ('" + sourceName + "')");
        return false;
    }
    // Don't collide with anything else that could be resolved as a source
    // name (a pass, group, or another scene texture).
    if (FindPass(sourceName) || FindGroup(sourceName) ||
        sceneTextures_.find(sourceName) != sceneTextures_.end()) {
        SetError("AddSceneTexture: name '" + sourceName + "' is already in use");
        return false;
    }

    RenderTexture2D tex;
    if (!AllocateRenderTarget(tex, width, height)) {
        SetError("AddSceneTexture: failed to allocate render target for '" + sourceName + "'");
        return false;
    }

    sceneTextures_[sourceName] = tex;
    return true;
}

bool RenderGraph::DestroySceneTexture(const std::string& sourceName) {
    auto it = sceneTextures_.find(sourceName);
    if (it == sceneTextures_.end()) {
        SetError("RemoveSceneTexture: no named scene texture '" + sourceName + "'");
        return false;
    }

    if (it->second.id != 0) UnloadRenderTexture(it->second);
    sceneTextures_.erase(it);

    // Clear any bindings that pointed at this now-gone source, mirroring
    // DestroyPass's cleanup of dangling sourcePassName references.
    for (ProcessPass& pass : passes_)
        for (TextureBinding& binding : pass.textureBindings)
            if (binding.sourcePassName == sourceName) binding.sourcePassName.clear();
    for (ComputePass& pass : computePasses_) {
        for (TextureBinding& binding : pass.textureBindings)
            if (binding.sourcePassName == sourceName) binding.sourcePassName.clear();
        for (ComputeImageBinding& binding : pass.imageBindings)
            if (binding.sourcePassName == sourceName) binding.sourcePassName.clear();
    }
    for (ProcessGroup& group : groups_)
        if (group.sourceName == sourceName) group.sourceName.clear();

    return true;
}

RenderTexture2D* RenderGraph::GetNamedSceneTexture(const std::string& sourceName) {
    auto it = sceneTextures_.find(sourceName);
    return it == sceneTextures_.end() ? nullptr : &it->second;
}

const RenderTexture2D* RenderGraph::GetNamedSceneTexture(const std::string& sourceName) const {
    auto it = sceneTextures_.find(sourceName);
    return it == sceneTextures_.end() ? nullptr : &it->second;
}

void RenderGraph::EnsureAllPassOutputsAllocated() {
    for (ProcessPass& pass : passes_) {
        if (pass.output.id != 0) continue;
        AllocatePassOutput(pass);
    }
}

ProcessGroup* RenderGraph::FindGroup(const std::string& name) {
    auto iterator = std::find_if(groups_.begin(), groups_.end(),
                                 [&](const ProcessGroup& group) { return group.name == name; });
    return iterator == groups_.end() ? nullptr : &(*iterator);
}

const ProcessGroup* RenderGraph::FindGroup(const std::string& name) const {
    auto iterator = std::find_if(groups_.begin(), groups_.end(),
                                 [&](const ProcessGroup& group) { return group.name == name; });
    return iterator == groups_.end() ? nullptr : &(*iterator);
}

bool RenderGraph::CreateGroupTemplate(const std::string& name) {
    if (name.empty() || FindGroupTemplate(name)) return false;

    ProcessGroupTemplate groupTemplate;
    groupTemplate.name = name;
    groupTemplates_.push_back(std::move(groupTemplate));
    return true;
}

bool RenderGraph::DestroyGroupTemplate(const std::string& name) {
    auto iterator = std::find_if(
        groupTemplates_.begin(),
        groupTemplates_.end(),
        [&](const ProcessGroupTemplate& groupTemplate) {
            return groupTemplate.name == name;
        });

    if (iterator == groupTemplates_.end()) return false;

    for (const ProcessGroupInstance& instance : groupInstances_)
        if (instance.templateName == name)
            return false;

    groupTemplates_.erase(iterator);
    return true;
}

bool RenderGraph::AttachPassToGroupTemplate(const std::string& templateName,
                                                const std::string& passName,
                                                const std::string& shaderName) {
    ProcessGroupTemplate* groupTemplate = FindGroupTemplate(templateName);
    if (!groupTemplate || passName.empty() || shaderName.empty()) return false;
    if (!FindShader(shaderName)) return false;

    auto iterator = std::find_if(
        groupTemplate->passes.begin(),
        groupTemplate->passes.end(),
        [&](const ProcessPassTemplate& passTemplate) {
            return passTemplate.name == passName;
        });

    if (iterator != groupTemplate->passes.end()) return false;

    groupTemplate->passes.push_back({passName, shaderName});
    return true;
}

GroupHandle RenderGraph::CreateGroupInstance(const std::string& instanceName,
                                                 const std::string& templateName) {
    if (width_ <= 0 || height_ <= 0 || instanceName.empty() ||
        instanceName == SceneSourceName() ||
        FindGroup(instanceName) || FindPass(instanceName) ||
        FindGroupInstance(instanceName)) {
        return GroupHandle();
    }

    const ProcessGroupTemplate* groupTemplate = FindGroupTemplate(templateName);
    if (!groupTemplate) return GroupHandle();

    if (!CreateProcessGroup(instanceName))
        return GroupHandle();

    ProcessGroupInstance instance;
    instance.name = instanceName;
    instance.templateName = templateName;
    instance.groupName = instanceName;

    groupInstances_.push_back(instance);

    for (const ProcessPassTemplate& passTemplate : groupTemplate->passes) {
        const std::string passName =
            instanceName + "/" + passTemplate.name;

        if (!CreatePass(passName, passTemplate.shaderName)) {
            DestroyProcessGroup(instanceName);
            groupInstances_.erase(
                std::remove_if(
                    groupInstances_.begin(),
                    groupInstances_.end(),
                    [&](const ProcessGroupInstance& value) {
                        return value.name == instanceName;
                    }),
                groupInstances_.end());
            return GroupHandle();
        }

        if (!AttachPassToGroup(instanceName, passName)) {
            DestroyPass(passName);
            DestroyProcessGroup(instanceName);
            groupInstances_.erase(
                std::remove_if(
                    groupInstances_.begin(),
                    groupInstances_.end(),
                    [&](const ProcessGroupInstance& value) {
                        return value.name == instanceName;
                    }),
                groupInstances_.end());
            return GroupHandle();
        }
    }

    return GroupHandle(instanceName);
}

bool RenderGraph::DestroyGroupInstance(const std::string& instanceName) {
    ProcessGroupInstance* instance = FindGroupInstance(instanceName);
    if (!instance) return false;

    ProcessGroup* group = FindGroup(instance->groupName);
    if (!group) {
        groupInstances_.erase(
            std::remove_if(
                groupInstances_.begin(),
                groupInstances_.end(),
                [&](const ProcessGroupInstance& value) {
                    return value.name == instanceName;
                }),
            groupInstances_.end());
        return false;
    }

    const std::vector<std::string> passNames = group->passNames;

    for (const std::string& passName : passNames)
        DestroyPass(passName);

    DestroyProcessGroup(instance->groupName);

    groupInstances_.erase(
        std::remove_if(
            groupInstances_.begin(),
            groupInstances_.end(),
            [&](const ProcessGroupInstance& value) {
                return value.name == instanceName;
            }),
        groupInstances_.end());

    return true;
}

ProcessGroupTemplate* RenderGraph::FindGroupTemplate(const std::string& name) {
    auto iterator = std::find_if(
        groupTemplates_.begin(),
        groupTemplates_.end(),
        [&](const ProcessGroupTemplate& groupTemplate) {
            return groupTemplate.name == name;
        });

    return iterator == groupTemplates_.end() ? nullptr : &(*iterator);
}

const ProcessGroupTemplate* RenderGraph::FindGroupTemplate(const std::string& name) const {
    auto iterator = std::find_if(
        groupTemplates_.begin(),
        groupTemplates_.end(),
        [&](const ProcessGroupTemplate& groupTemplate) {
            return groupTemplate.name == name;
        });

    return iterator == groupTemplates_.end() ? nullptr : &(*iterator);
}

ProcessGroupInstance* RenderGraph::FindGroupInstance(const std::string& name) {
    auto iterator = std::find_if(
        groupInstances_.begin(),
        groupInstances_.end(),
        [&](const ProcessGroupInstance& instance) {
            return instance.name == name;
        });

    return iterator == groupInstances_.end() ? nullptr : &(*iterator);
}

const ProcessGroupInstance* RenderGraph::FindGroupInstance(const std::string& name) const {
    auto iterator = std::find_if(
        groupInstances_.begin(),
        groupInstances_.end(),
        [&](const ProcessGroupInstance& instance) {
            return instance.name == name;
        });

    return iterator == groupInstances_.end() ? nullptr : &(*iterator);
}

ProcessGroup* RenderGraph::FindGroupContainingPass(const std::string& passName) {
    for (ProcessGroup& group : groups_)
        if (std::find(group.passNames.begin(), group.passNames.end(), passName) != group.passNames.end())
            return &group;

    return nullptr;
}

const ProcessGroup* RenderGraph::FindGroupContainingPass(const std::string& passName) const {
    for (const ProcessGroup& group : groups_)
        if (std::find(group.passNames.begin(), group.passNames.end(), passName) != group.passNames.end())
            return &group;

    return nullptr;
}

bool RenderGraph::IsPassGrouped(const std::string& passName) const {
    return FindGroupContainingPass(passName) != nullptr;
}

bool RenderGraph::AttachPassToGroup(const std::string& groupName, const std::string& passName) {
    ProcessGroup* group = FindGroup(groupName);
    ProcessPass* pass = FindPass(passName);

    if (!group || !pass) return false;
    if (FindGroupContainingPass(passName)) return false;

    group->passNames.push_back(passName);
    return true;
}

bool RenderGraph::RemovePassFromGroup(const std::string& groupName, const std::string& passName) {
    ProcessGroup* group = FindGroup(groupName);
    if (!group) return false;

    auto iterator = std::find(group->passNames.begin(), group->passNames.end(), passName);
    if (iterator == group->passNames.end()) return false;

    group->passNames.erase(iterator);
    return true;
}

bool RenderGraph::SetGroupEnabled(const std::string& groupName, bool enabled) {
    ProcessGroup* group = FindGroup(groupName);
    if (!group) return false;

    group->enabled = enabled;
    return true;
}

bool RenderGraph::SetGroupSource(const std::string& groupName, const std::string& sourceName) {
    ProcessGroup* group = FindGroup(groupName);
    if (!group) return false;

    if (sourceName.empty()) {
        group->sourceName.clear();
        return true;
    }

    if (sourceName == SceneSourceName()) {
        group->sourceName = sourceName;
        return true;
    }

    if (sourceName == groupName) return false;

    if (!FindPass(sourceName) && !FindGroup(sourceName)) return false;

    const ProcessGroup* sourceGroup = FindGroup(sourceName);
    if (sourceGroup) {
        group->sourceName = sourceName;
        return true;
    }

    const ProcessPass* sourcePass = FindPass(sourceName);
    if (!sourcePass) return false;

    if (!sourcePass->persistentOutput && !previewAllPasses_)
        return false;

    group->sourceName = sourceName;
    return true;
}

bool RenderGraph::SetEnabled(const std::string& passName, bool enabled) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    pass->enabled = enabled;
    return true;
}
bool RenderGraph::SetPassOutputFormat(const std::string& passName, int format, bool withDepth) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    if (pass->outputFormat == format && pass->outputWithDepth == withDepth)
        return true;

    int oldFormat = pass->outputFormat;
    bool oldWithDepth = pass->outputWithDepth;

    if (pass->output.id != 0) {
        ReleaseToPool(pass->output, oldFormat, oldWithDepth);
        pass->output = {};
    }

    pass->outputFormat = format;
    pass->outputWithDepth = withDepth;

    if (pass->output.id == 0 && (pass->persistentOutput || previewAllPasses_)) {
        if (!AllocatePassOutput(*pass))
            return false;
    }

    return true;
}
bool RenderGraph::SetPassClass(const std::string& passName, const std::string& className) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;
    pass->classNames.clear();
    if (!className.empty())
        pass->classNames.push_back(className);
    return true;
}
bool RenderGraph::SetPassTargetFramebuffer(const std::string& passName, const std::string& framebufferName) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;
    if (!framebufferName.empty() && !FindFramebuffer(framebufferName)) return false;   // must exist
    pass->targetFramebufferName = framebufferName;
    return true;
}

bool RenderGraph::SetPersistentOutput(const std::string& passName, bool persistent) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    if (pass->persistentOutput == persistent) return true;

    if (persistent) {
        if (pass->output.id == 0 && !AllocatePassOutput(*pass))
            return false;

        pass->persistentOutput = true;
    } else {
        pass->persistentOutput = false;
        if (!previewAllPasses_ && pass->output.id != 0) {
            ReleaseToPool(pass->output, pass->outputFormat, pass->outputWithDepth);
            pass->output = {};
        }
    }

    return true;
}

bool RenderGraph::SetResolutionScale(const std::string& passName, float scale) {
    ProcessPass* pass = FindPass(passName);
    if (!pass || scale <= 0.0f) return false;

    if (!pass->persistentOutput && !previewAllPasses_)
        return false;

    pass->resolutionScale = scale;

    if (!AllocatePassOutput(*pass))
        return false;

    if (pass->enableFeedback) {
        RenderTexture2D newFeedback;

        if (!AllocateScaledRenderTarget(
            newFeedback,
            width_,
            height_,
            pass->resolutionScale,
            pass->outputFilter)) {
            return false;
        }

        ClearRenderTarget(newFeedback, BLANK);

        if (pass->feedback.id != 0)
            UnloadRenderTexture(pass->feedback);

        pass->feedback = newFeedback;
    }

    return true;
}

bool RenderGraph::SetOutputFilter(const std::string& passName, TextureFilter filter) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    pass->outputFilter = filter;

    if (pass->output.id != 0)
        SetTextureFilter(pass->output.texture, filter);

    if (pass->feedback.id != 0)
        SetTextureFilter(pass->feedback.texture, filter);

    return true;
}

bool RenderGraph::SetFeedbackEnabled(const std::string& passName, bool enabled) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    if (pass->enableFeedback == enabled) return true;

    if (enabled) {
        RenderTexture2D newFeedback;

        if (!AllocateScaledRenderTarget(
            newFeedback,
            width_,
            height_,
            pass->resolutionScale,
            pass->outputFilter)) {
            return false;
        }

        ClearRenderTarget(newFeedback, BLANK);

        if (pass->feedback.id != 0)
            UnloadRenderTexture(pass->feedback);

        pass->feedback = newFeedback;
        pass->enableFeedback = true;
    } else {
        pass->enableFeedback = false;

        if (pass->feedback.id != 0) {
            UnloadRenderTexture(pass->feedback);
            pass->feedback = {};
        }
    }

    return true;
}

bool RenderGraph::SetPassHistory(const std::string& passName, bool enable) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) {
        SetError("SetPassHistory: no pass named '" + passName + "'");
        return false;
    }

    if (pass->keepHistory == enable) return true;

    if (enable) {
        // Size the history buffer to match the pass's own output settings
        // (same scale/filter as pass.output), not the current pass.output
        // allocation, since output may not be allocated yet (e.g. pass
        // isn't persistentOutput / preview isn't on).
        RenderTexture2D newHistory;
        if (!AllocateScaledRenderTarget(newHistory, width_, height_, pass->resolutionScale, pass->outputFilter)) {
            SetError("SetPassHistory: failed to allocate history buffer for '" + passName + "'");
            return false;
        }
        ClearRenderTarget(newHistory, BLANK);

        if (pass->history.id != 0)
            UnloadRenderTexture(pass->history);

        pass->history = newHistory;
        pass->keepHistory = true;
    } else {
        pass->keepHistory = false;
        if (pass->history.id != 0) {
            UnloadRenderTexture(pass->history);
            pass->history = {};
        }
    }

    return true;
}

const Texture2D* RenderGraph::GetPassHistoryTexture(const std::string& passName) const {
    const ProcessPass* pass = FindPass(passName);
    if (!pass || !pass->keepHistory || pass->history.id == 0) return nullptr;
    return &pass->history.texture;
}

void RenderGraph::SwapHistory(ProcessPass& pass, Texture2D sourceTexture) {
    // sourceTexture is passed in explicitly (rather than read from
    // pass.output) because a pass without a dedicated output -- i.e. not
    // persistentOutput and previewAllPasses_ is off -- renders into a
    // shared ping-pong buffer instead, so pass.output.id would be 0 even
    // though the pass produced a perfectly good frame this call.
    if (!pass.keepHistory || pass.history.id == 0 || sourceTexture.id == 0) return;

    // Copy this frame's freshly-rendered output into the history buffer so
    // next frame's HistorySourceName() lookup returns *this* frame's result
    // (i.e. "previous frame" from next frame's point of view).
    BeginTextureMode(pass.history);
    ClearBackground(BLANK);
    DrawTexturePro(
        sourceTexture,
        FlippedSourceRectangle(sourceTexture),
        FullTargetRectangle(pass.history.texture.width, pass.history.texture.height),
        Vector2Zero(), 0.0f, WHITE);
    EndTextureMode();
}

bool RenderGraph::SetPassViewport(const std::string& passName, Rectangle rect) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) { SetError("SetPassViewport: no pass named '" + passName + "'"); return false; }
    pass->viewport = rect;
    return true;
}

bool RenderGraph::SetPassScissor(const std::string& passName, Rectangle rect) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) { SetError("SetPassScissor: no pass named '" + passName + "'"); return false; }
    pass->scissor = rect;
    return true;
}

bool RenderGraph::SetComputeViewport(const std::string& passName, Rectangle rect) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) { SetError("SetComputeViewport: no compute pass named '" + passName + "'"); return false; }
    // Compute passes don't rasterize, but the rect is stored for API
    // symmetry / potential future use. No direct GL effect here.
    (void)rect;
    return true;
}

bool RenderGraph::SetComputeScissor(const std::string& passName, Rectangle rect) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) { SetError("SetComputeScissor: no compute pass named '" + passName + "'"); return false; }
    (void)rect;
    return true;
}

bool RenderGraph::SetGeometryViewport(const std::string& passName, Rectangle rect) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) { SetError("SetGeometryViewport: no geometry pass named '" + passName + "'"); return false; }
    pass->viewport = rect;
    return true;
}

bool RenderGraph::SetGeometryScissor(const std::string& passName, Rectangle rect) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) { SetError("SetGeometryScissor: no geometry pass named '" + passName + "'"); return false; }
    pass->scissor = rect;
    return true;
}

bool RenderGraph::SetPassGenerateMipmaps(const std::string& passName, bool enable) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) { SetError("SetPassGenerateMipmaps: no pass named '" + passName + "'"); return false; }
    pass->generateMipmaps = enable;
    return true;
}

bool RenderGraph::SetPassBlendMode(const std::string& passName, PassBlendMode mode, int customBlendMode) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) { SetError("SetPassBlendMode: no pass named '" + passName + "'"); return false; }
    pass->renderState.blendMode = mode;
    pass->renderState.customBlendMode = customBlendMode;
    return true;
}

bool RenderGraph::SetPassDepthTest(const std::string& passName, bool enabled, bool writeEnabled) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) { SetError("SetPassDepthTest: no pass named '" + passName + "'"); return false; }
    pass->renderState.depthTest = enabled;
    pass->renderState.depthWrite = writeEnabled;
    return true;
}

bool RenderGraph::SetGeometryBlendMode(const std::string& passName, PassBlendMode mode, int customBlendMode) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) { SetError("SetGeometryBlendMode: no geometry pass named '" + passName + "'"); return false; }
    pass->renderState.blendMode = mode;
    pass->renderState.customBlendMode = customBlendMode;
    return true;
}

bool RenderGraph::SetGeometryDepthTest(const std::string& passName, bool enabled, bool writeEnabled) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) { SetError("SetGeometryDepthTest: no geometry pass named '" + passName + "'"); return false; }
    pass->renderState.depthTest = enabled;
    pass->renderState.depthWrite = writeEnabled;
    return true;
}

// Applies a raster pass's blend/depth/stencil state before its draw call.
// Blending defaults to a hard overwrite (matches the old unconditional
// rlDisableColorBlend()); depth/stencil test are opt-in per pass so
// ProcessPass compositing (no depth buffer) is unaffected, while
// GeometryPass defaults to depth-on (see CreateGeometryPass).
void RenderGraph::ApplyPassRenderState(const PassRenderState& state) {
    if (state.blendMode == PassBlendMode::Custom) {
        BeginBlendMode(state.customBlendMode);
    } else {
        rlDisableColorBlend();
    }

    if (state.depthTest) {
        rlEnableDepthTest();
        if (state.depthWrite) rlEnableDepthMask();
        else rlDisableDepthMask();
    } else {
        rlDisableDepthTest();
    }
}

void RenderGraph::RestorePassRenderState(const PassRenderState& state) {
    if (state.blendMode == PassBlendMode::Custom) {
        EndBlendMode();
    } else {
        rlEnableColorBlend();
    }

    if (state.depthTest) rlDisableDepthTest();
}

bool RenderGraph::SetPassPriority(const std::string& passName, int priority) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) { SetError("SetPassPriority: no pass named '" + passName + "'"); return false; }
    pass->priority = priority;
    return true;
}

bool RenderGraph::SetComputePriority(const std::string& passName, int priority) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) { SetError("SetComputePriority: no compute pass named '" + passName + "'"); return false; }
    pass->priority = priority;
    return true;
}

bool RenderGraph::SetGeometryPriority(const std::string& passName, int priority) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) { SetError("SetGeometryPriority: no geometry pass named '" + passName + "'"); return false; }
    pass->priority = priority;
    return true;
}

bool RenderGraph::SetComputePersistentState(const std::string& passName, bool enable) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) { SetError("SetComputePersistentState: no compute pass named '" + passName + "'"); return false; }
    pass->persistentState = enable;
    return true;
}

int RenderGraph::GetNodePriority(const std::string& name) const {
    if (const ProcessPass* p = FindPass(name)) return p->priority;
    if (const ComputePass* p = FindComputePass(name)) return p->priority;
    if (const GeometryPass* p = FindGeometryPass(name)) return p->priority;
    if (const ProcessGroup* g = FindGroup(name)) return 0; // groups have no priority
    if (const ComputeGroup* g = FindComputeGroup(name)) return g->priority;
    return 0;
}

// ---- Direct uniform setters ----
// These fire outside the main Apply() loop (e.g. a UI slider changing a
// value between frames), so unlike ExecutePass/ReapplyUniforms they own
// their own enable/disable pair -- there's no larger "already active
// shader" context to piggyback on here.
bool RenderGraph::SetFloat(const std::string& passName, const std::string& uniformName, float value) {
    ProcessPass* pass = FindPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_FLOAT, currentShaderId_);
}

bool RenderGraph::SetInt(const std::string& passName, const std::string& uniformName, int value) {
    ProcessPass* pass = FindPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_INT, currentShaderId_);
}

bool RenderGraph::SetVector2(const std::string& passName, const std::string& uniformName, Vector2 value) {
    ProcessPass* pass = FindPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC2, currentShaderId_);
}

bool RenderGraph::SetVector3(const std::string& passName, const std::string& uniformName, Vector3 value) {
    ProcessPass* pass = FindPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC3, currentShaderId_);
}

bool RenderGraph::SetVector4(const std::string& passName, const std::string& uniformName, Vector4 value) {
    ProcessPass* pass = FindPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC4, currentShaderId_);
}

bool RenderGraph::SetColor(const std::string& passName, const std::string& uniformName, Color value) {
    ProcessPass* pass = FindPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    // Colors are stored/compared as the raw 0-255 Color -- normalization
    // to a 0-1 Vector4 happens only at the point of the GPU push, inside
    // ApplyCachedUniform, so the cache and ApplyCachedUniform's equality
    // check both see the same value the caller passed in.
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC4, currentShaderId_);
}

bool RenderGraph::SetComputeFloat(const std::string& passName, const std::string& uniformName, float value) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_FLOAT, currentShaderId_);
}

bool RenderGraph::SetComputeInt(const std::string& passName, const std::string& uniformName, int value) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_INT, currentShaderId_);
}

bool RenderGraph::SetComputeVector2(const std::string& passName, const std::string& uniformName, Vector2 value) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC2, currentShaderId_);
}

bool RenderGraph::SetComputeVector3(const std::string& passName, const std::string& uniformName, Vector3 value) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC3, currentShaderId_);
}

bool RenderGraph::SetComputeVector4(const std::string& passName, const std::string& uniformName, Vector4 value) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC4, currentShaderId_);
}

bool RenderGraph::SetComputeColor(const std::string& passName, const std::string& uniformName, Color value) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC4, currentShaderId_);
}
bool RenderGraph::SetClassFloat(const std::string& className,
                                const std::string& uniformName,
                                float value) {
    bool found = false;

    // Post-process passes
    for (ProcessPass& pass : passes_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetFloat(pass.name, uniformName, value);
    }

    // Compute passes
    for (ComputePass& pass : computePasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetComputeFloat(pass.name, uniformName, value);
    }

    // Geometry passes
    for (GeometryPass& pass : geometryPasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetGeometryFloat(pass.name, uniformName, value);
    }

    return found;
}

bool RenderGraph::SetClassInt(const std::string& className,
                              const std::string& uniformName,
                              int value) {
    bool found = false;

    for (ProcessPass& pass : passes_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetInt(pass.name, uniformName, value);
    }

    for (ComputePass& pass : computePasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetComputeInt(pass.name, uniformName, value);
    }

    for (GeometryPass& pass : geometryPasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetGeometryInt(pass.name, uniformName, value);
    }

    return found;
}

bool RenderGraph::SetClassVector2(const std::string& className,
                                  const std::string& uniformName,
                                  Vector2 value) {
    bool found = false;

    for (ProcessPass& pass : passes_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetVector2(pass.name, uniformName, value);
    }

    for (ComputePass& pass : computePasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetComputeVector2(pass.name, uniformName, value);
    }

    for (GeometryPass& pass : geometryPasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetGeometryVector2(pass.name, uniformName, value);
    }

    return found;
}

bool RenderGraph::SetClassVector3(const std::string& className,
                                  const std::string& uniformName,
                                  Vector3 value) {
    bool found = false;

    for (ProcessPass& pass : passes_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetVector3(pass.name, uniformName, value);
    }

    for (ComputePass& pass : computePasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetComputeVector3(pass.name, uniformName, value);
    }

    for (GeometryPass& pass : geometryPasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetGeometryVector3(pass.name, uniformName, value);
    }

    return found;
}

bool RenderGraph::SetClassVector4(const std::string& className,
                                  const std::string& uniformName,
                                  Vector4 value) {
    bool found = false;

    for (ProcessPass& pass : passes_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetVector4(pass.name, uniformName, value);
    }

    for (ComputePass& pass : computePasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetComputeVector4(pass.name, uniformName, value);
    }

    for (GeometryPass& pass : geometryPasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetGeometryVector4(pass.name, uniformName, value);
    }

    return found;
}

bool RenderGraph::SetClassColor(const std::string& className,
                                const std::string& uniformName,
                                Color value) {
    bool found = false;

    for (ProcessPass& pass : passes_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetColor(pass.name, uniformName, value);
    }

    for (ComputePass& pass : computePasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetComputeColor(pass.name, uniformName, value);
    }

    for (GeometryPass& pass : geometryPasses_) {
        if (std::find(pass.classNames.begin(), pass.classNames.end(), className) == pass.classNames.end())
            continue;
        found = true;
        SetGeometryColor(pass.name, uniformName, value);
    }

    return found;
}
PassTextureBindingHandle RenderGraph::CreateTextureBinding(const std::string& passName,
                                           const std::string& bindingName,
                                           const std::string& uniformName) {
    ProcessPass* pass = FindPass(passName);
    if (!pass || !pass->shaderDefinition || bindingName.empty() || uniformName.empty()) return PassTextureBindingHandle();

    const bool duplicateName = FindTextureBinding(*pass, bindingName) != nullptr;

    const bool duplicateUniform = std::any_of(
        pass->textureBindings.begin(),
        pass->textureBindings.end(),
        [&](const TextureBinding& binding) {
            return binding.uniformName == uniformName;
        }
    );

    if (duplicateName || duplicateUniform) return PassTextureBindingHandle();

    const int location = rlGetLocationUniform(pass->shaderDefinition->shader.id, uniformName.c_str());
    if (location < 0) return PassTextureBindingHandle();

    pass->textureBindings.push_back(
        {bindingName, uniformName, Texture2D{}, location, {}, {}}
    );

    return PassTextureBindingHandle(PassHandle(passName), bindingName);
}

bool RenderGraph::SetTexture(const std::string& passName,
                                    const std::string& bindingName,
                                    Texture2D texture,
                                    const std::string& texturePath) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    TextureBinding* binding = FindTextureBinding(*pass, bindingName);
    if (!binding) return false;

    binding->texture = texture;
    binding->sourcePassName.clear();
    // A binding can only be one kind at a time -- setting a plain 2D
    // texture here means it's no longer whatever it was before (e.g. if
    // it had previously been pointed at a cubemap/array framebuffer via
    // SetTextureSource, or at a 3D texture via SetTexture3D below).
    binding->isCubemap = false;
    binding->isTextureArray = false;
    binding->isTexture3D = false;

    if (!texturePath.empty() || texture.id == 0)
        binding->texturePath = texturePath;

    return true;
}

bool RenderGraph::SetTexture3D(const std::string& passName,
                                      const std::string& bindingName,
                                      Texture2D texture,
                                      const std::string& texturePath) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    TextureBinding* binding = FindTextureBinding(*pass, bindingName);
    if (!binding) return false;

    binding->texture = texture;
    binding->sourcePassName.clear();
    binding->isCubemap = false;
    binding->isTextureArray = false;
    binding->isTexture3D = true;

    if (!texturePath.empty() || texture.id == 0)
        binding->texturePath = texturePath;

    return true;
}

Texture2D RenderGraph::LoadTexture3DFromAtlas(const std::string& path, int tilesX, int tilesY) {
    if (path.empty() || tilesX <= 0 || tilesY <= 0) return (Texture2D){ 0 };

    Image atlas = LoadImage(path.c_str());
    if (atlas.data == nullptr) {
        TraceLog(LOG_WARNING, "LoadTexture3DFromAtlas: failed to load %s", path.c_str());
        return (Texture2D){ 0 };
    }
    if (atlas.width % tilesX != 0 || atlas.height % tilesY != 0) {
        TraceLog(LOG_WARNING,
            "LoadTexture3DFromAtlas: %s is %dx%d, not evenly divisible by %dx%d tiles",
            path.c_str(), atlas.width, atlas.height, tilesX, tilesY);
        UnloadImage(atlas);
        return (Texture2D){ 0 };
    }

    // Normalize to a known, tightly-packed format so the deinterleave below
    // can index raw bytes directly -- same format PixelFormatToGL is asked
    // to map, so keep the two in sync if this ever grows more formats.
    ImageFormat(&atlas, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    unsigned int internalFormat, format, type;
    if (!PixelFormatToGL(RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, internalFormat, format, type)) {
        UnloadImage(atlas);
        return (Texture2D){ 0 };
    }

    const int tileW  = atlas.width / tilesX;
    const int tileH  = atlas.height / tilesY;
    const int layers = tilesX * tilesY;
    const int bpp    = 4; // R8G8B8A8

    // Atlas is laid out tilesX across, tilesY down; glTexImage3D wants the
    // data layer-major (all of layer 0's texels, then layer 1's, ...).
    // NOTE: verified against the reference Dither3D_2x2.png -- the tile at
    // the visual TOP of the file is the densest/last layer (dotCount ==
    // dotsTotal), and the visual BOTTOM tile is the sparsest/first layer
    // (dotCount == 1). Since LoadImage/stb_image reads rows top-down (row
    // 0 == visual top), a naive top-to-bottom tile scan would assign the
    // densest tile to GL layer 0 -- backwards from what the shader's
    // subLayer math expects (Z=0 -> sparsest, Z=layers-1 -> densest). So
    // the row of tiles is walked top-down for reading, but written to the
    // layer index counting up from the BOTTOM tile.
    const unsigned char* src = static_cast<const unsigned char*>(atlas.data);
    std::vector<unsigned char> volume(static_cast<size_t>(tileW) * tileH * layers * bpp);

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            int layer = (tilesY - 1 - ty) * tilesX + tx;
            for (int y = 0; y < tileH; ++y) {
                int srcY = ty * tileH + y;
                const unsigned char* srcRow = src + (static_cast<size_t>(srcY) * atlas.width + static_cast<size_t>(tx) * tileW) * bpp;
                unsigned char* dstRow = volume.data() + ((static_cast<size_t>(layer) * tileH + y) * tileW) * bpp;
                memcpy(dstRow, srcRow, static_cast<size_t>(tileW) * bpp);
            }
        }
    }
    UnloadImage(atlas);

    // No rlgl path for volume textures (see SetTexture3D's comment in the
    // header) -- raw GL for the object itself, same as
    // CreateTextureArrayFramebuffer's GL_TEXTURE_2D_ARRAY path. Still route
    // the unit select through rlActiveTextureSlot so this doesn't disturb
    // rlgl's own texture-unit bookkeeping.
    unsigned int texId = 0;
    glGenTextures(1, &texId);
    if (texId == 0) return (Texture2D){ 0 };

    rlActiveTextureSlot(0);
    glBindTexture(GL_TEXTURE_3D, texId);
    glTexImage3D(GL_TEXTURE_3D, 0, static_cast<int>(internalFormat), tileW, tileH, layers,
                 0, format, type, volume.data());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // Layers are selected explicitly by the shader (no blending across the
    // first/last layer wanted), so clamp rather than repeat on the Z axis.
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_3D, 0);

    Texture2D tex = { 0 };
    tex.id      = texId;
    tex.width   = tileW;
    tex.height  = tileH;
    tex.mipmaps = 1;
    tex.format  = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    return tex;
}

bool RenderGraph::SetTextureSource(const std::string& passName,
                                          const std::string& bindingName,
                                          const std::string& sourceName) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    TextureBinding* binding = FindTextureBinding(*pass, bindingName);
    if (!binding) return false;

    if (sourceName.empty()) {
        binding->sourcePassName.clear();
        return true;
    }

    if (sourceName == SceneSourceName() || sceneTextures_.find(sourceName) != sceneTextures_.end()) {
        binding->sourcePassName = sourceName;
        binding->texture = {};
        return true;
    }

    if (sourceName == HistorySourceName()) {
        // History is per-pass; this binds to *this* pass's own history
        // (equivalent to the feedback self-reference below, but reads the
        // previous *frame's* output rather than a same-frame copy).
        if (!pass->keepHistory) return false;
        binding->sourcePassName = sourceName;
        binding->texture = {};
        return true;
    }

    if (sourceName == pass->name) {
        if (!pass->enableFeedback) return false;

        binding->sourcePassName = sourceName;
        binding->texture = {};
        return true;
    }

    ProcessGroup* sourceGroup = FindGroup(sourceName);
    if (sourceGroup) {
        binding->sourcePassName = sourceName;
        binding->texture = {};
        return true;
    }

    // A framebuffer name -- this is how a GeometryPass's rendered output
    // (or several GeometryPasses sharing one target) reaches a ProcessPass.
    // No persistentOutput/previewAllPasses_ gate here: unlike a ProcessPass
    // output, a Framebuffer always owns its texture for the graph's
    // lifetime, so there's nothing extra to opt into.
    if (const Framebuffer* fb = FindFramebuffer(sourceName)) {
        binding->sourcePassName = sourceName;
        binding->texture = {};
        binding->isCubemap = fb->isCubemap;
        binding->isTextureArray = fb->isTextureArray;
        binding->framebufferAttachmentIndex = 0;
        return true;
    }

    const ProcessPass* sourcePass = FindPass(sourceName);
    if (!sourcePass) return false;

    if (!sourcePass->persistentOutput && !previewAllPasses_)
        return false;

    binding->sourcePassName = sourceName;
    binding->texture = {};
    return true;
}

bool RenderGraph::SetTextureSourceFramebufferAttachment(const std::string& passName,
                                                          const std::string& bindingName,
                                                          const std::string& framebufferName,
                                                          std::size_t attachmentIndex) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    TextureBinding* binding = FindTextureBinding(*pass, bindingName);
    if (!binding) return false;

    const Framebuffer* fb = FindFramebuffer(framebufferName);
    if (!fb || attachmentIndex >= fb->colorAttachments.size()) return false;

    binding->sourcePassName = framebufferName;
    binding->texture = {};
    binding->isCubemap = fb->isCubemap;
    binding->isTextureArray = fb->isTextureArray;
    binding->framebufferAttachmentIndex = attachmentIndex;
    return true;
}

bool RenderGraph::DestroyTextureBinding(const std::string& passName,
                                              const std::string& bindingName) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    auto iterator = std::find_if(
        pass->textureBindings.begin(),
        pass->textureBindings.end(),
        [&](const TextureBinding& binding) {
            return binding.name == bindingName;
        }
    );

    if (iterator == pass->textureBindings.end()) return false;

    pass->textureBindings.erase(iterator);
    return true;
}

bool RenderGraph::ClearTextureBindings(const std::string& passName) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;

    pass->textureBindings.clear();
    return true;
}

// -----------------------------------------------------------------------------------
// Geometry pass texture (sampler) binding functions
// -----------------------------------------------------------------------------------
// Same shape as the ProcessPass functions above, minus a SetGeometryTextureSource
// equivalent -- geometry passes only take a directly-supplied Texture2D/raw id
// for now, not "read from another pass's output."

GeometryTextureBindingHandle RenderGraph::CreateGeometryTextureBinding(const std::string& passName,
                                                                        const std::string& bindingName,
                                                                        const std::string& uniformName) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass || !pass->shaderDefinition || bindingName.empty() || uniformName.empty())
        return GeometryTextureBindingHandle();

    const bool duplicateName = FindTextureBinding(*pass, bindingName) != nullptr;

    const bool duplicateUniform = std::any_of(
        pass->textureBindings.begin(),
        pass->textureBindings.end(),
        [&](const TextureBinding& binding) {
            return binding.uniformName == uniformName;
        }
    );

    if (duplicateName || duplicateUniform) return GeometryTextureBindingHandle();

    const int location = rlGetLocationUniform(pass->shaderDefinition->shader.id, uniformName.c_str());
    if (location < 0) return GeometryTextureBindingHandle();

    pass->textureBindings.push_back(
        {bindingName, uniformName, Texture2D{}, location, {}, {}}
    );

    return GeometryTextureBindingHandle(GeometryPassHandle(passName), bindingName);
}

bool RenderGraph::SetGeometryTexture(const std::string& passName,
                                      const std::string& bindingName,
                                      Texture2D texture,
                                      const std::string& texturePath) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;

    TextureBinding* binding = FindTextureBinding(*pass, bindingName);
    if (!binding) return false;

    binding->texture = texture;
    binding->sourcePassName.clear();
    binding->isCubemap = false;
    binding->isTextureArray = false;
    binding->isTexture3D = false;

    if (!texturePath.empty() || texture.id == 0)
        binding->texturePath = texturePath;

    return true;
}

bool RenderGraph::SetGeometryTexture3D(const std::string& passName,
                                        const std::string& bindingName,
                                        Texture2D texture,
                                        const std::string& texturePath) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;

    TextureBinding* binding = FindTextureBinding(*pass, bindingName);
    if (!binding) return false;

    binding->texture = texture;
    binding->sourcePassName.clear();
    binding->isCubemap = false;
    binding->isTextureArray = false;
    binding->isTexture3D = true;

    if (!texturePath.empty() || texture.id == 0)
        binding->texturePath = texturePath;

    return true;
}

bool RenderGraph::DestroyGeometryTextureBinding(const std::string& passName,
                                                 const std::string& bindingName) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;

    auto iterator = std::find_if(
        pass->textureBindings.begin(),
        pass->textureBindings.end(),
        [&](const TextureBinding& binding) {
            return binding.name == bindingName;
        }
    );

    if (iterator == pass->textureBindings.end()) return false;

    pass->textureBindings.erase(iterator);
    return true;
}

// -----------------------------------------------------------------------------------
// Compute texture (sampler) binding functions
// -----------------------------------------------------------------------------------

ComputeTextureBindingHandle RenderGraph::CreateComputeTextureBinding(const std::string& passName,
                                                  const std::string& bindingName,
                                                  const std::string& uniformName) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass || !pass->programDefinition || bindingName.empty() || uniformName.empty())
        return ComputeTextureBindingHandle();

    // Check for duplicate name or uniform
    for (const auto& binding : pass->textureBindings) {
        if (binding.name == bindingName || binding.uniformName == uniformName)
            return ComputeTextureBindingHandle();
    }

    int location = rlGetLocationUniform(pass->programDefinition->id, uniformName.c_str());
    if (location < 0) return ComputeTextureBindingHandle();

    TextureBinding binding;
    binding.name = bindingName;
    binding.uniformName = uniformName;
    binding.location = location;

    pass->textureBindings.push_back(binding);
    return ComputeTextureBindingHandle(ComputePassHandle(passName), bindingName);
}

bool RenderGraph::SetComputeTextureRegistrySource(const std::string& passName,
                                                    const std::string& bindingName,
                                                    const std::string& registryName) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass || registryName.empty()) return false;

    for (auto& binding : pass->textureBindings) {
        if (binding.name == bindingName) {
            binding.kernelRegistryName = registryName;
            // These become irrelevant while kernelRegistryName is set (see
            // the resolution order in DispatchComputePass), but clear them
            // anyway so a later inspection of the binding isn't confusing.
            binding.sourcePassName.clear();
            binding.texture = {};
            return true;
        }
    }
    return false;
}

bool RenderGraph::SetComputeTexture(const std::string& passName,
                                           const std::string& bindingName,
                                           Texture2D texture) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;

    for (auto& binding : pass->textureBindings) {
        if (binding.name == bindingName) {
            binding.texture = texture;
            binding.sourcePassName.clear(); // clear any source reference
            binding.kernelRegistryName.clear(); // explicit texture overrides a registry link
            return true;
        }
    }
    return false;
}
bool RenderGraph::SetComputeImageRawTexture(const std::string& passName,
                                            const std::string& bindingName,
                                            unsigned int textureId,
                                            unsigned int glInternalFormat) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    for (auto& binding : pass->imageBindings) {
        if (binding.name == bindingName) {
            binding.texture.id = textureId;
            binding.glInternalFormat = glInternalFormat;
            binding.isRawTexture = true;
            binding.sourcePassName.clear();
            binding.kernelRegistryName.clear(); // explicit raw texture overrides a registry link
            return true;
        }
    }
    return false;
}
bool RenderGraph::SetComputeTextureSource(const std::string& passName,
                                                 const std::string& bindingName,
                                                 const std::string& sourceName) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;

    for (auto& binding : pass->textureBindings) {
        if (binding.name == bindingName) {
            binding.kernelRegistryName.clear(); // explicit source overrides a registry link
            if (sourceName.empty()) {
                binding.sourcePassName.clear();
                binding.texture = {};
                return true;
            }

            // Validate source exists
            if (sourceName == SceneSourceName() || sceneTextures_.find(sourceName) != sceneTextures_.end()) {
                binding.sourcePassName = sourceName;
                binding.texture = {};
                return true;
            }

            const ProcessPass* srcPass = FindPass(sourceName);
            const ProcessGroup* srcGroup = FindGroup(sourceName);
            const Framebuffer* srcFb = FindFramebuffer(sourceName);
            if (!srcPass && !srcGroup && !srcFb) return false;

            // Store source; resolution happens at dispatch time.
            binding.sourcePassName = sourceName;
            binding.texture = {};
            return true;
        }
    }
    return false;
}

bool RenderGraph::DestroyComputeTextureBinding(const std::string& passName,
                                                     const std::string& bindingName) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;

    auto it = std::find_if(pass->textureBindings.begin(), pass->textureBindings.end(),
                           [&](const TextureBinding& b) { return b.name == bindingName; });
    if (it == pass->textureBindings.end()) return false;

    pass->textureBindings.erase(it);
    return true;
}

bool RenderGraph::ClearComputeTextureBindings(const std::string& passName) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;

    pass->textureBindings.clear();
    return true;
}
ComputeImageBindingHandle RenderGraph::CreateComputeImageBinding(const std::string& passName,
                                                const std::string& bindingName,
                                                const std::string& uniformName,
                                                bool readOnly,
                                                int format,
                                                int imageUnit) {   // default -1 = auto assign
    ComputePass* pass = FindComputePass(passName);
    if (!pass || !pass->programDefinition || bindingName.empty() || uniformName.empty())
        return ComputeImageBindingHandle();

    // Check duplicates
    for (const auto& b : pass->imageBindings) {
        if (b.name == bindingName || b.uniformName == uniformName)
            return ComputeImageBindingHandle();
        if (imageUnit >= 0 && b.imageUnit == static_cast<unsigned int>(imageUnit))
            return ComputeImageBindingHandle(); // unit already used
    }

    if (imageUnit < 0) {
        // Find first free unit
        imageUnit = 0;
        for (const auto& b : pass->imageBindings) {
            if (b.imageUnit == static_cast<unsigned int>(imageUnit)) {
                imageUnit++;
                // restart search from beginning
                for (const auto& b2 : pass->imageBindings) {
                    if (b2.imageUnit == static_cast<unsigned int>(imageUnit)) {
                        imageUnit++;
                        break;
                    }
                }
            }
        }
    }

    int location = rlGetLocationUniform(pass->programDefinition->id, uniformName.c_str());
    if (location < 0) return ComputeImageBindingHandle();

    ComputeImageBinding binding;
    binding.name = bindingName;
    binding.uniformName = uniformName;
    binding.location = location;
    binding.readOnly = readOnly;
    binding.format = format;
    binding.imageUnit = static_cast<unsigned int>(imageUnit);

    pass->imageBindings.push_back(binding);
    return ComputeImageBindingHandle(ComputePassHandle(passName), bindingName);
}
bool RenderGraph::SetComputeImageTexture(const std::string& passName,
                                                const std::string& bindingName,
                                                Texture2D texture) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;

    for (auto& binding : pass->imageBindings) {
        if (binding.name == bindingName) {
            binding.texture = texture;
            binding.sourcePassName.clear(); // clear any source reference
            binding.kernelRegistryName.clear(); // explicit texture overrides a registry link
            return true;
        }
    }
    return false;
}
bool RenderGraph::SetComputeImageSource(const std::string& passName,
                                               const std::string& bindingName,
                                               const std::string& sourceName) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;

    for (auto& binding : pass->imageBindings) {
        if (binding.name == bindingName) {
            binding.kernelRegistryName.clear(); // explicit source overrides a registry link
            if (sourceName.empty()) {
                binding.sourcePassName.clear();
                binding.texture = {};
                return true;
            }

            // Validate source exists (basic check)
            if (sourceName == SceneSourceName() || sceneTextures_.find(sourceName) != sceneTextures_.end()) {
                binding.sourcePassName = sourceName;
                binding.texture = {};
                return true;
            }

            // Check if it's a pass, group, or framebuffer that can provide a texture
            const ProcessPass* srcPass = FindPass(sourceName);
            const ProcessGroup* srcGroup = FindGroup(sourceName);
            const Framebuffer* srcFb = FindFramebuffer(sourceName);
            if (!srcPass && !srcGroup && !srcFb) return false;

            // For now, just store the name; actual resolution happens at dispatch.
            binding.sourcePassName = sourceName;
            binding.texture = {};
            return true;
        }
    }
    return false;
}

bool RenderGraph::SetComputeImageRegistrySource(const std::string& passName,
                                                  const std::string& bindingName,
                                                  const std::string& registryName) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass || registryName.empty()) return false;

    for (auto& binding : pass->imageBindings) {
        if (binding.name == bindingName) {
            binding.kernelRegistryName = registryName;
            binding.sourcePassName.clear();
            binding.texture = {};
            return true;
        }
    }
    return false;
}

bool RenderGraph::DestroyComputeImageBinding(const std::string& passName,
                                                   const std::string& bindingName) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;

    auto it = std::find_if(pass->imageBindings.begin(), pass->imageBindings.end(),
                           [&](const ComputeImageBinding& b) { return b.name == bindingName; });
    if (it == pass->imageBindings.end()) return false;

    pass->imageBindings.erase(it);
    return true;
}

bool RenderGraph::ClearComputeImageBindings(const std::string& passName) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    pass->imageBindings.clear();
    return true;
}

bool RenderGraph::CopyComputeImageBindings(const std::string& fromPassName, const std::string& toPassName) {
    ComputePass* from = FindComputePass(fromPassName);
    ComputePass* to = FindComputePass(toPassName);
    if (!from || !to || from == to) return false;

    bool anyCopied = false;
    for (const ComputeImageBinding& src : from->imageBindings) {
        // Re-derives location/imageUnit against toPass's own program --
        // this is not a struct copy. Skips (doesn't abort) on a
        // name/uniform collision so one pre-existing binding on toPass
        // doesn't block the rest from copying.
        ComputeImageBindingHandle handle =
            CreateComputeImageBinding(toPassName, src.name, src.uniformName, src.readOnly, src.format);
        if (!handle) continue;

        if (src.isRawTexture) {
            SetComputeImageRawTexture(toPassName, src.name, src.texture.id, src.glInternalFormat);
        } else if (!src.kernelRegistryName.empty()) {
            SetComputeImageRegistrySource(toPassName, src.name, src.kernelRegistryName);
        } else if (!src.sourcePassName.empty()) {
            SetComputeImageSource(toPassName, src.name, src.sourcePassName);
        } else if (src.texture.id != 0) {
            SetComputeImageTexture(toPassName, src.name, src.texture);
        }
        anyCopied = true;
    }
    return anyCopied;
}

bool RenderGraph::CopyComputeTextureBindings(const std::string& fromPassName, const std::string& toPassName) {
    ComputePass* from = FindComputePass(fromPassName);
    ComputePass* to = FindComputePass(toPassName);
    if (!from || !to || from == to) return false;

    bool anyCopied = false;
    for (const TextureBinding& src : from->textureBindings) {
        ComputeTextureBindingHandle handle = CreateComputeTextureBinding(toPassName, src.name, src.uniformName);
        if (!handle) continue;

        if (!src.kernelRegistryName.empty()) {
            SetComputeTextureRegistrySource(toPassName, src.name, src.kernelRegistryName);
        } else if (!src.sourcePassName.empty()) {
            SetComputeTextureSource(toPassName, src.name, src.sourcePassName);
        } else if (src.texture.id != 0) {
            SetComputeTexture(toPassName, src.name, src.texture);
        }
        anyCopied = true;
    }
    return anyCopied;
}

bool RenderGraph::CopyComputeBufferBindings(const std::string& fromPassName, const std::string& toPassName) {
    ComputePass* from = FindComputePass(fromPassName);
    ComputePass* to = FindComputePass(toPassName);
    if (!from || !to || from == to) return false;

    bool anyCopied = false;
    for (const ComputeBufferBinding& src : from->bufferBindings) {
        if (BindBufferToComputePass(toPassName, src.bufferName, src.bindingIndex))
            anyCopied = true;
    }
    return anyCopied;
}

bool RenderGraph::CopyAllComputeBindings(const std::string& fromPassName, const std::string& toPassName) {
    ComputePass* from = FindComputePass(fromPassName);
    ComputePass* to = FindComputePass(toPassName);
    if (!from || !to || from == to) return false;

    // Deliberately not short-circuited (a | b | c, not a || b || c) so a
    // false from one category doesn't skip the others.
    bool images = CopyComputeImageBindings(fromPassName, toPassName);
    bool textures = CopyComputeTextureBindings(fromPassName, toPassName);
    bool buffers = CopyComputeBufferBindings(fromPassName, toPassName);

    // Uniform values, too -- pushed through the normal Set* path (so
    // toPass's own ApplyCachedUniform dirty-check still applies) rather
    // than copied as raw struct data, since a raw copy would carry over
    // fromPass's cached GL location, which is meaningless on toPass.
    bool uniforms = false;
    for (const Uniform& u : from->uniforms) {
        bool set = std::visit([&](auto&& v) -> bool {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, float>) return SetComputeFloat(toPassName, u.name, v);
            else if constexpr (std::is_same_v<T, int>) return SetComputeInt(toPassName, u.name, v);
            else if constexpr (std::is_same_v<T, Vector2>) return SetComputeVector2(toPassName, u.name, v);
            else if constexpr (std::is_same_v<T, Vector3>) return SetComputeVector3(toPassName, u.name, v);
            else if constexpr (std::is_same_v<T, Vector4>) return SetComputeVector4(toPassName, u.name, v);
            else if constexpr (std::is_same_v<T, Color>) return SetComputeColor(toPassName, u.name, v);
            else return false; // monostate: never actually set, nothing to copy
        }, u.value);
        uniforms = uniforms || set;
    }

    return images || textures || buffers || uniforms;
}

void RenderGraph::ReapplyUniforms(ProcessPass& pass) {
    // NOTE: assumes rlEnableShader(pass.shaderDefinition->shader.id) has
    // already been called by the caller. This function does not enable or
    // disable the shader itself -- see the header comment for why (the
    // caller usually has more work to do, like texture binding and
    // drawing, that also needs the shader to stay active).
    if (!pass.shaderDefinition || pass.shaderDefinition->shader.id == 0)
        return;

    for (Uniform& uniform : pass.uniforms) {
        ApplyUniformRLGL(uniform.location, uniform.value);
    }
}
void RenderGraph::ReapplyComputeUniforms(ComputePass& pass) {
    if (!pass.programDefinition || pass.programDefinition->id == 0)
        return;

    for (Uniform& uniform : pass.uniforms) {
        ApplyUniformRLGL(uniform.location, uniform.value);
    }
}
bool RenderGraph::ReloadPassShader(const std::string& passName) {
    ProcessPass* pass = FindPass(passName);
    return pass ? ReloadShader(pass->shaderName) : false;
}

bool RenderGraph::ReloadShader(const std::string& shaderName) {
    auto it = std::find_if(shaders_.begin(), shaders_.end(),
        [&](const ProcessShader& shader) { return shader.name == shaderName; });
    if (it == shaders_.end()) return false;

    Shader newShader = LoadShader(it->vertexShaderPath.empty() ? nullptr : it->vertexShaderPath.c_str(),
                                  it->fragmentShaderPath.c_str());
    if (newShader.id == 0) return false;
    if (it->shader.id != 0) UnloadShader(it->shader);
    it->shader = newShader;

    for (ProcessPass& pass : passes_) {
        if (pass.shaderName != shaderName) continue;
        pass.shaderDefinition = &(*it);
        for (Uniform& uniform : pass.uniforms)
            uniform.location = rlGetLocationUniform(it->shader.id, uniform.name.c_str());
        for (TextureBinding& binding : pass.textureBindings)
            binding.location = rlGetLocationUniform(it->shader.id, binding.uniformName.c_str());

        // ReapplyUniforms assumes an already-active shader (see its
        // comment), so this call site owns its own enable/disable pair.
        rlEnableShader(pass.shaderDefinition->shader.id);
        ReapplyUniforms(pass);
        rlDisableShader();
        currentShaderId_ = 0;
    }
    return true;
}
bool RenderGraph::ReloadComputeShader(const std::string& shaderName) {
    auto it = std::find_if(computePrograms_.begin(), computePrograms_.end(),
        [&](const ComputeProgram& program) { return program.name == shaderName; });
    if (it == computePrograms_.end()) return false;

    // Load the compute shader source from the same path used originally.
    char* source = LoadFileText(it->sourcePath.c_str());
    if (source == nullptr) return false;

    unsigned int csId = rlLoadShader(source, RL_COMPUTE_SHADER);
    UnloadFileText(source);

    if (csId == 0) return false;

    unsigned int newId = rlLoadShaderProgramCompute(csId);
    rlUnloadShader(csId);
    if (newId == 0) return false;

    // Replace the old program.
    if (it->id != 0) rlUnloadShaderProgram(it->id);
    it->id = newId;

    // Update cached uniform locations for all passes using this program.
    for (ComputePass& pass : computePasses_) {
        if (pass.programName != shaderName) continue;
        pass.programDefinition = &(*it);
        for (Uniform& u : pass.uniforms)
            u.location = rlGetLocationUniform(newId, u.name.c_str());
        // Optional: reapply uniforms immediately if the pass will be dispatched soon.
        // ReapplyComputeUniforms(*pass); // requires shader enabled; caller can do it.
    }
    return true;
}
void RenderGraph::Apply() {
    if (sceneTexture_.id == 0) {
        currentOutput_ = {};
        return;
    }
    // Automatic uniforms: push anything SetUniform() changed since each
    // kernel pass last saw it. No explicit ApplyTo() call needed -- this
    // is the one place per frame it needs to happen.
    ApplyGlobalUniformsToKernelPasses();
    computeWrittenTextures_.clear();
    barrierFlushedTextures_.clear();
    if (previewAllPasses_)
        EnsureAllPassOutputsAllocated();

    Texture2D input = sceneTexture_.texture;
    int pingIndex = 0;
    std::unordered_set<std::string> executedPasses;
    std::unordered_set<std::string> executedGroups;
    std::vector<RenderNode> executionOrder;
    if (automaticOrdering_) {
        ComputeExecutionOrder(executionOrder);
    } else {
        executionOrder = renderOrder_;
    }

    // ---- Profiling: read previous frame results ----
    if (profilingEnabled_) {
        for (auto& pair : queryIds_) {
            GLuint64 elapsed = 0;
            glGetQueryObjectui64v(pair.second, GL_QUERY_RESULT, &elapsed);
            gpuTimes_[pair.first] = static_cast<float>(elapsed) / 1e6f; // ns -> ms
        }
        // Reset times for nodes that will execute this frame
        for (const auto& node : executionOrder) {
            gpuTimes_[node.name] = 0.0f;
        }
    }

    // Minimal liveness pass (unchanged)
    std::unordered_map<std::string, std::size_t> lastUseIndex;
    if (previewAllPasses_) {
        for (std::size_t i = 0; i < executionOrder.size(); ++i) {
            const RenderNode& node = executionOrder[i];
            auto noteUse = [&](const std::string& src) {
                if (src.empty() || src == SceneSourceName() || src == HistorySourceName()) return;
                lastUseIndex[src] = i;
            };
            if (node.type == RenderNodeType::Pass) {
                const ProcessPass* p = FindPass(node.name);
                if (p) for (const TextureBinding& b : p->textureBindings) noteUse(b.sourcePassName);
            } 
            else if (node.type == RenderNodeType::Compute) {
                const ComputePass* p = FindComputePass(node.name);
                if (p) {
                    for (const TextureBinding& b : p->textureBindings) noteUse(b.sourcePassName);
                    for (const ComputeImageBinding& b : p->imageBindings) noteUse(b.sourcePassName);
                }
                
            }
            else if (node.type == RenderNodeType::Group) {
                const ProcessGroup* g = FindGroup(node.name);
                if (g) noteUse(g->sourceName);
            }
        }
    }
    auto ReleaseIfNoLongerNeeded = [&](std::size_t currentIndex) {
        if (!previewAllPasses_) return;
        for (ProcessPass& p : passes_) {
            if (p.persistentOutput || p.output.id == 0) continue;
            auto it = lastUseIndex.find(p.name);
            std::size_t deadline = (it != lastUseIndex.end()) ? it->second
                : [&] {
                    auto selfIt = std::find_if(executionOrder.begin(), executionOrder.end(),
                        [&](const RenderNode& n) { return n.type == RenderNodeType::Pass && n.name == p.name; });
                    return selfIt == executionOrder.end() ? currentIndex
                                                           : static_cast<std::size_t>(selfIt - executionOrder.begin());
                }();
            if (deadline <= currentIndex) {
                ReleaseToPool(p.output, p.outputFormat, p.outputWithDepth);
                p.output = {};
            }
        }
    };

    auto ResolveSource = [&](const std::string& sourceName) -> Texture2D {
        if (sourceName.empty())
            return Texture2D{};

        if (sourceName == SceneSourceName())
            return sceneTexture_.texture;

        {
            auto namedScene = sceneTextures_.find(sourceName);
            if (namedScene != sceneTextures_.end())
                return namedScene->second.texture;
        }

        if (executedGroups.find(sourceName) != executedGroups.end()) {
            const ProcessGroup* group = FindGroup(sourceName);
            if (group && group->output.id != 0)
                return group->output.texture;
        }

        if (executedPasses.find(sourceName) != executedPasses.end()) {
            const ProcessPass* pass = FindPass(sourceName);

            if (pass &&
                (pass->persistentOutput || previewAllPasses_) &&
                pass->output.id != 0) {
                return pass->output.texture;
            }
        }

        // Named framebuffer -- this is how GeometryPass output reaches the
        // rest of the graph. GeometryPass renders into a Framebuffer
        // (possibly several passes sharing one, e.g. opaque -> water ->
        // foliage via clearTarget=false), not into a ProcessPass-style
        // output texture, so it's resolved by the framebuffer's own name
        // rather than any single pass's name -- unambiguous no matter how
        // many passes wrote into it this frame.
        {
            Texture2D fbTexture = ResolveFramebufferColorTexture(sourceName);
            if (fbTexture.id != 0) return fbTexture;
        }

        return Texture2D{};
    };

        auto ExecutePass = [&](ProcessPass& pass, Texture2D passInput) -> Texture2D {
            if (!pass.enabled || !pass.shaderDefinition || pass.shaderDefinition->shader.id == 0)
                return passInput;

            const bool wantsDedicated = pass.persistentOutput || previewAllPasses_;
            const bool useDedicated = wantsDedicated && pass.output.id != 0;

            // If a custom framebuffer is specified, render to it and return input unchanged
            if (!pass.targetFramebufferName.empty()) {
                Framebuffer* fb = FindFramebuffer(pass.targetFramebufferName);
                if (!fb || fb->id == 0) return passInput;

                rlEnableFramebuffer(fb->id);
                rlActiveDrawBuffers(static_cast<int>(fb->colorAttachments.size()));
                rlClearColor(pass.clearColor.r, pass.clearColor.g, pass.clearColor.b, pass.clearColor.a);
                rlClearScreenBuffers();

                ApplyPassRenderState(pass.renderState);

                rlEnableShader(pass.shaderDefinition->shader.id);
                ReapplyUniforms(pass);

                for (const auto& binding : pass.uniformBufferBindings) {
                    UniformBuffer* buffer = FindUniformBuffer(binding.bufferName);
                    if (buffer && buffer->id != 0) {
                        glBindBufferBase(GL_UNIFORM_BUFFER, binding.bindingIndex, buffer->id);
                    }
                }

                int textureSlot = 1;
                for (TextureBinding& binding : pass.textureBindings) {
                    Texture2D boundTexture = binding.texture;

                    if (!binding.sourcePassName.empty()) {
                        if (binding.sourcePassName == HistorySourceName()) {
                            boundTexture =
                                (pass.keepHistory && pass.history.id != 0)
                                ? pass.history.texture
                                : Texture2D{};
                        } else if (binding.sourcePassName == pass.name) {
                            boundTexture =
                                (pass.enableFeedback && pass.feedback.id != 0)
                                ? pass.feedback.texture
                                : Texture2D{};
                        } else {
                            Texture2D sourceTexture = ResolveSource(binding.sourcePassName);
                            if (sourceTexture.id != 0)
                                boundTexture = sourceTexture;
                            else
                                boundTexture = Texture2D{};
                        }
                    }

                    if (binding.location >= 0 && boundTexture.id != 0) {
                        BindTextureToSamplerTyped(binding, boundTexture.id, textureSlot);
                        textureSlot++;

                        // Ensure compute writes to this texture are visible to the sampler
                        EnsureBarrierForTexture(boundTexture.id);
                    }
                }

                const bool fbHasViewport = pass.viewport.width > 0 && pass.viewport.height > 0;
                const bool fbHasScissor = pass.scissor.width > 0 && pass.scissor.height > 0;
                if (fbHasViewport) {
                    rlViewport(static_cast<int>(pass.viewport.x), static_cast<int>(pass.viewport.y),
                               static_cast<int>(pass.viewport.width), static_cast<int>(pass.viewport.height));
                }
                if (fbHasScissor) {
                    rlEnableScissorTest();
                    rlScissor(static_cast<int>(pass.scissor.x), static_cast<int>(pass.scissor.y),
                              static_cast<int>(pass.scissor.width), static_cast<int>(pass.scissor.height));
                }

                if (pass.useClipSpaceQuad) {
                    rlLoadDrawQuad();
                } else {
                    DrawTexturePro(passInput,
                        FlippedSourceRectangle(passInput),
                        FullTargetRectangle(fb->width, fb->height),
                        Vector2Zero(), 0.0f, WHITE);
                }
                rlDrawRenderBatchActive();

                if (fbHasScissor) rlDisableScissorTest();
                if (fbHasViewport) rlViewport(0, 0, fb->width, fb->height);

                RestorePassRenderState(pass.renderState);
                rlDisableShader();
                currentShaderId_ = 0;
                rlDisableFramebuffer();

                return passInput;   // chain unchanged
            }

            RenderTexture2D& target =
                useDedicated ? pass.output : pingPongBuffers_[pingIndex];

            if (target.id == 0)
                return passInput;

            BeginTextureMode(target);
            ClearBackground(pass.clearColor);

            rlEnableShader(pass.shaderDefinition->shader.id);

            ReapplyUniforms(pass);
            for (const auto& binding : pass.uniformBufferBindings) {
                UniformBuffer* buffer = FindUniformBuffer(binding.bufferName);
                if (buffer && buffer->id != 0) {
                    glBindBufferBase(GL_UNIFORM_BUFFER, binding.bindingIndex, buffer->id);
                }
            }

            int textureSlot = 1;
            for (TextureBinding& binding : pass.textureBindings) {
                Texture2D boundTexture = binding.texture;

                if (!binding.sourcePassName.empty()) {
                    if (binding.sourcePassName == HistorySourceName()) {
                        boundTexture =
                            (pass.keepHistory && pass.history.id != 0)
                            ? pass.history.texture
                            : Texture2D{};
                    } else if (binding.sourcePassName == pass.name) {
                        boundTexture =
                            (pass.enableFeedback && pass.feedback.id != 0)
                            ? pass.feedback.texture
                            : Texture2D{};
                    } else {
                        Texture2D sourceTexture = ResolveSource(binding.sourcePassName);
                        if (sourceTexture.id != 0)
                            boundTexture = sourceTexture;
                        else
                            boundTexture = Texture2D{};
                    }
                }

                if (binding.location >= 0 && boundTexture.id != 0) {
                    BindTextureToSamplerTyped(binding, boundTexture.id, textureSlot);
                    textureSlot++;

                    // Ensure compute writes to this texture are visible to the sampler
                    EnsureBarrierForTexture(boundTexture.id);
                }
            }

            ApplyPassRenderState(pass.renderState);

            const bool hasViewport = pass.viewport.width > 0 && pass.viewport.height > 0;
            const bool hasScissor = pass.scissor.width > 0 && pass.scissor.height > 0;
            if (hasViewport) {
                rlViewport(static_cast<int>(pass.viewport.x), static_cast<int>(pass.viewport.y),
                           static_cast<int>(pass.viewport.width), static_cast<int>(pass.viewport.height));
            }
            if (hasScissor) {
                rlEnableScissorTest();
                rlScissor(static_cast<int>(pass.scissor.x), static_cast<int>(pass.scissor.y),
                          static_cast<int>(pass.scissor.width), static_cast<int>(pass.scissor.height));
            }

            if (pass.useClipSpaceQuad) {
                rlLoadDrawQuad();
            } else {
                DrawTexturePro(
                    passInput,
                    FlippedSourceRectangle(passInput),
                    FullTargetRectangle(target.texture.width, target.texture.height),
                    Vector2Zero(),
                    0.0f,
                    WHITE
                );
            }

            rlDrawRenderBatchActive();

            if (hasScissor) rlDisableScissorTest();
            if (hasViewport) rlViewport(0, 0, target.texture.width, target.texture.height);

            RestorePassRenderState(pass.renderState);
            rlDisableShader();
            currentShaderId_ = 0;
            EndTextureMode();

            if (pass.generateMipmaps && target.texture.id != 0) {
                int mipmapCount = 0;
                rlGenTextureMipmaps(target.texture.id, target.texture.width,
                                     target.texture.height, target.texture.format, &mipmapCount);
            }

            Texture2D result = target.texture;

            if (!useDedicated)
                pingIndex = 1 - pingIndex;

            if (pass.enableFeedback && pass.feedback.id != 0) {
                BeginTextureMode(pass.feedback);
                ClearBackground(BLANK);

                DrawTexturePro(
                    target.texture,
                    FlippedSourceRectangle(target.texture),
                    FullTargetRectangle(
                        pass.feedback.texture.width,
                        pass.feedback.texture.height
                    ),
                    Vector2Zero(),
                    0.0f,
                    WHITE
                );

                EndTextureMode();
            }

            if (pass.keepHistory)
                SwapHistory(pass, result);

            executedPasses.insert(pass.name);
            return result;
        };
    

    // Helper to create query scope for a node
    auto createQueryScope = [&](const std::string& nodeName) -> QueryScope {
        GLuint qid = 0;
        if (profilingEnabled_) {
            auto it = queryIds_.find(nodeName);
            if (it == queryIds_.end()) {
                glGenQueries(1, &qid);
                queryIds_[nodeName] = qid;
            } else {
                qid = it->second;
            }
        }
        return QueryScope(profilingEnabled_, qid);
    };

    for (std::size_t nodeIndex = 0; nodeIndex < executionOrder.size(); ++nodeIndex) {
        const RenderNode& node = executionOrder[nodeIndex];
        if (node.type == RenderNodeType::Pass) {
            ProcessPass* pass = FindPass(node.name);
            if (!pass) continue;
            if (pass->condition && !pass->condition()) continue;
            if (IsPassGrouped(pass->name))
                continue;

            auto queryScope = createQueryScope(node.name);
            input = ExecutePass(*pass, input);
        }
        else if (node.type == RenderNodeType::Group) {
            ProcessGroup* group = FindGroup(node.name);
            if (!group || !group->enabled)
                continue;
            if (group->condition && !group->condition()) continue;

            auto queryScope = createQueryScope(node.name);

            Texture2D groupInput = input;

            if (!group->sourceName.empty()) {
                Texture2D sourceTexture = ResolveSource(group->sourceName);
                if (sourceTexture.id != 0)
                    groupInput = sourceTexture;
            }

            for (const std::string& passName : group->passNames) {
                ProcessPass* pass = FindPass(passName);
                if (!pass) continue;

                groupInput = ExecutePass(*pass, groupInput);
            }

            if (group->output.id == 0)
                AllocateGroupOutput(*group);

            if (group->output.id == 0)
                continue;

            BeginTextureMode(group->output);
            ClearBackground(BLANK);

            DrawTexturePro(
                groupInput,
                FlippedSourceRectangle(groupInput),
                FullTargetRectangle(
                    group->output.texture.width,
                    group->output.texture.height
                ),
                Vector2Zero(),
                0.0f,
                WHITE
            );

            EndTextureMode();

            input = group->output.texture;
            executedGroups.insert(group->name);
        }
        else if (node.type == RenderNodeType::Compute) {
            ComputePass* computePass = FindComputePass(node.name);
            if (!computePass) continue;
            if (computePass->condition && !computePass->condition()) continue;

            auto queryScope = createQueryScope(node.name);
            DispatchComputePass(computePass->name);
        }
        else if (node.type == RenderNodeType::ComputeGroup) {
            ComputeGroup* group = FindComputeGroup(node.name);
            if (!group || !group->enabled) continue;
            if (group->condition && !group->condition()) continue;

            auto queryScope = createQueryScope(node.name);

            for (const std::string& passName : group->computePassNames) {
                ComputePass* pass = FindComputePass(passName);
                if (!pass || !pass->enabled) continue;
                if (pass->condition && !pass->condition()) continue;

                DispatchComputePass(pass->name);
            }
        }
        else if (node.type == RenderNodeType::Geometry) {
            GeometryPass* geoPass = FindGeometryPass(node.name);
            if (!geoPass || !geoPass->enabled || !geoPass->shaderDefinition) continue;
            if (geoPass->condition && !geoPass->condition()) continue;

            Framebuffer* fb = FindFramebuffer(geoPass->targetFramebufferName);
            if (!fb || fb->id == 0) continue;
            auto queryScope = createQueryScope(node.name);
            rlEnableFramebuffer(fb->id);
            rlActiveDrawBuffers(static_cast<int>(fb->colorAttachments.size()));

            ApplyPassRenderState(geoPass->renderState);

            const bool geoHasViewport = geoPass->viewport.width > 0 && geoPass->viewport.height > 0;
            const bool geoHasScissor = geoPass->scissor.width > 0 && geoPass->scissor.height > 0;
            if (geoHasScissor) {
                rlEnableScissorTest();
                rlScissor(static_cast<int>(geoPass->scissor.x), static_cast<int>(geoPass->scissor.y),
                          static_cast<int>(geoPass->scissor.width), static_cast<int>(geoPass->scissor.height));
            }

            rlEnableShader(geoPass->shaderDefinition->shader.id);
            for (auto& uniform : geoPass->uniforms) {
                ApplyUniformRLGL(uniform.location, uniform.value);
            }
            // mat4 uniforms set via SetGeometryMatrix -- see MatrixUniform's
            // doc comment for why these don't go through ApplyUniformRLGL.
            for (auto& matrixUniform : geoPass->matrixUniforms) {
                if (matrixUniform.location < 0) continue;
                rlSetUniformMatrix(matrixUniform.location, matrixUniform.value);
            }
            for (const auto& binding : geoPass->uniformBufferBindings) {
                UniformBuffer* buffer = FindUniformBuffer(binding.bufferName);
                if (buffer && buffer->id != 0) {
                    glBindBufferBase(GL_UNIFORM_BUFFER, binding.bindingIndex, buffer->id);
                }
            }
            for (const auto& binding : geoPass->bufferBindings) {
                const ShaderBuffer* buffer = FindBuffer(binding.bufferName);
                if (buffer && buffer->id != 0) {
                    rlBindShaderBuffer(buffer->id, binding.bindingIndex);
                }
            }

            // Sampler bindings (see CreateGeometryTextureBinding/
            // SetGeometryTexture/SetGeometryTexture3D) -- same isCubemap/
            // isTextureArray/isTexture3D dispatch as the ProcessPass and
            // ComputePass binding loops. No sourcePassName resolution here
            // since geometry passes don't support SetGeometryTextureSource
            // yet -- binding.texture is always the directly-set texture.
            int geoTextureSlot = 1;
            for (const TextureBinding& binding : geoPass->textureBindings) {
                if (binding.location < 0 || binding.texture.id == 0) continue;

                EnsureBarrierForTexture(binding.texture.id);

                rlActiveTextureSlot(geoTextureSlot);
                if (binding.isCubemap) {
                    glBindTexture(GL_TEXTURE_CUBE_MAP, binding.texture.id);
                } else if (binding.isTextureArray) {
                    glBindTexture(GL_TEXTURE_2D_ARRAY, binding.texture.id);
                } else if (binding.isTexture3D) {
                    glBindTexture(GL_TEXTURE_3D, binding.texture.id);
                } else {
                    rlEnableTexture(binding.texture.id);
                }
                int unit = geoTextureSlot;
                rlSetUniform(binding.location, &unit, RL_SHADER_UNIFORM_INT, 1);
                geoTextureSlot++;
            }

            // Runs drawItemsProvider() (if set) right after drawCallback and
            // issues one glDrawArrays per returned GeometryDrawItem, binding
            // that item's buffers to their SSBO slots first. Called once per
            // drawCallback invocation below -- which, for a cubemap-capture
            // pass, means once per face. See GeometryDrawItem's doc comment
            // for why this loop never registers a persistent binding.
            auto executeGeometryDrawItems = [&]() {
                if (!geoPass->drawItemsProvider) return;

                if (geoPass->attributelessDraw) {
                    rlEnableVertexArray(attributelessVAO_);
                    if (geoPass->primitiveMode == PrimitiveMode::Points) rlEnablePointMode();
                }

                const unsigned int glPrimitive = ToGLPrimitive(geoPass->primitiveMode);
                for (const GeometryDrawItem& item : geoPass->drawItemsProvider()) {
                    for (const auto& [bindingIndex, bufferName] : item.bufferBindings) {
                        const ShaderBuffer* buf = FindBuffer(bufferName);
                        if (buf && buf->id != 0) rlBindShaderBuffer(buf->id, bindingIndex);
                    }
                    // No rlgl equivalent for a plain "draw N vertices from
                    // whatever's currently bound" call -- rlgl's own draw
                    // helpers are built around its immediate-mode batching,
                    // not an externally-bound SSBO -- so this is raw GL the
                    // same way glFramebufferTextureLayer is above.
                    glDrawArrays(glPrimitive, static_cast<int>(item.firstVertex), static_cast<int>(item.vertexCount));
                }

                if (geoPass->attributelessDraw) {
                    if (geoPass->primitiveMode == PrimitiveMode::Points) rlDisablePointMode();
                    rlDisableVertexArray();
                }
            };

            if (geoPass->isCubemapCapture && geoPass->cubemapCamera && fb->isCubemap) {
                // Standard OpenGL cubemap face convention -- target
                // direction and up vector per face, ordered +X/-X/+Y/-Y/
                // +Z/-Z to match RL_ATTACHMENT_CUBEMAP_POSITIVE_X + face
                // below. Get this table wrong and faces come out rotated
                // or mirrored rather than cleanly failing, so if the
                // capture looks "scrambled", check here first.
                static const Vector3 kFaceTargets[6] = {
                    { 1.0f,  0.0f,  0.0f}, {-1.0f,  0.0f,  0.0f},
                    { 0.0f,  1.0f,  0.0f}, { 0.0f, -1.0f,  0.0f},
                    { 0.0f,  0.0f,  1.0f}, { 0.0f,  0.0f, -1.0f},
                };
                static const Vector3 kFaceUps[6] = {
                    {0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
                    {0.0f,  0.0f, 1.0f}, {0.0f,  0.0f, -1.0f},
                    {0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
                };

                rlViewport(0, 0, fb->faceSize, fb->faceSize);
                // Verify rlSetClipPlanes exists in your rlgl.h -- name/
                // signature has moved around between raylib versions.
                // Harmless to drop if it doesn't: rlgl's default clip
                // planes will apply instead.
                rlSetClipPlanes(geoPass->cubemapNearPlane, geoPass->cubemapFarPlane);

                for (int face = 0; face < 6; ++face) {
                    for (const auto& att : fb->colorAttachments) {
                        rlFramebufferAttach(fb->id, att.textureId, att.attachType,
                                             RL_ATTACHMENT_CUBEMAP_POSITIVE_X + face, 0);
                    }

                    rlClearColor(geoPass->clearColor.r, geoPass->clearColor.g, geoPass->clearColor.b, geoPass->clearColor.a);
                    rlClearScreenBuffers();

                    geoPass->cubemapCamera->position = geoPass->cubemapProbePosition;
                    geoPass->cubemapCamera->target = Vector3Add(geoPass->cubemapProbePosition, kFaceTargets[face]);
                    geoPass->cubemapCamera->up = kFaceUps[face];
                    geoPass->cubemapCamera->fovy = 90.0f;
                    geoPass->cubemapCamera->projection = CAMERA_PERSPECTIVE;

                    // Same callback shape as a non-cubemap GeometryPass --
                    // BeginMode3D(*cubemapCamera)/draw/EndMode3D(). It has
                    // no idea it's being called 6 times with a mutated
                    // camera instead of once.
                    if (geoPass->drawCallback) geoPass->drawCallback();
                    executeGeometryDrawItems();
                }
            } else if (fb->isTextureArray) {
                // No forced camera/projection here, unlike the cubemap
                // branch above -- a texture-array slice is just "the
                // N'th independent image," so there's no shared math to
                // own on the graph's behalf. The one thing the graph
                // still has to do is point the FBO at the right slice,
                // since rlFramebufferAttach's texType enum has no
                // array-layer option -- glFramebufferTextureLayer is the
                // raw-GL call for that, same as glTexImage3D was the only
                // way to allocate the array in CreateTextureArrayFramebuffer.
                for (size_t i = 0; i < fb->colorAttachments.size(); ++i) {
                    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + static_cast<int>(i),
                                               fb->colorAttachments[i].textureId, 0, geoPass->targetArrayLayer);
                }

                if (geoPass->clearTarget) {
                    rlClearColor(geoPass->clearColor.r, geoPass->clearColor.g, geoPass->clearColor.b, geoPass->clearColor.a);
                    rlClearScreenBuffers();
                }

                if (geoHasViewport) {
                    rlViewport(static_cast<int>(geoPass->viewport.x), static_cast<int>(geoPass->viewport.y),
                               static_cast<int>(geoPass->viewport.width), static_cast<int>(geoPass->viewport.height));
                } else {
                    rlViewport(0, 0, fb->width, fb->height);
                }

                // Single draw, into whichever layer targetArrayLayer
                // currently names -- call SetGeometryPassTargetArrayLayer
                // between Apply() calls (or use one GeometryPass per
                // layer) to fill the other slices.
                if (geoPass->drawCallback) geoPass->drawCallback();
                executeGeometryDrawItems();
            } else {
                // Clear is skipped when chaining onto a target another pass
                // in this frame already drew into -- see
                // GeometryPass::clearTarget.
                if (geoPass->clearTarget) {
                    rlClearColor(geoPass->clearColor.r, geoPass->clearColor.g, geoPass->clearColor.b, geoPass->clearColor.a);
                    rlClearScreenBuffers();
                }

                if (geoHasViewport) {
                    rlViewport(static_cast<int>(geoPass->viewport.x), static_cast<int>(geoPass->viewport.y),
                               static_cast<int>(geoPass->viewport.width), static_cast<int>(geoPass->viewport.height));
                }

                if (geoPass->drawCallback) geoPass->drawCallback();
                executeGeometryDrawItems();
            }

            if (geoHasScissor) rlDisableScissorTest();
            if (geoHasViewport || geoPass->isCubemapCapture || fb->isTextureArray) rlViewport(0, 0, fb->width, fb->height);

            rlDisableShader();
            currentShaderId_ = 0;
            RestorePassRenderState(geoPass->renderState);
            rlDisableFramebuffer();
        }

        ReleaseIfNoLongerNeeded(nodeIndex);
    }

    currentOutput_ = input;
}

void RenderGraph::DrawOutput(Rectangle destination,
                                    Color tint,
                                    int blendMode) const {
    if (currentOutput_.id == 0) return;

    const bool customBlend = blendMode >= 0;

    // Same reasoning as the rlDisableColorBlend() in ExecutePass: without
    // an explicit blend mode requested, this should be a direct overwrite
    // of the destination, not a blend against whatever alpha currentOutput_
    // happens to carry. Leaving the global blend state untouched here meant
    // this draw silently used whatever was left on from the last pass
    // (usually re-enabled), re-introducing the exact same "washed out
    // toward the clear color" problem one step downstream of ExecutePass.
    if (!customBlend)
        rlDisableColorBlend();
    else
        BeginBlendMode(blendMode);

    DrawTexturePro(
        currentOutput_,
        FlippedSourceRectangle(currentOutput_),
        destination,
        Vector2Zero(),
        0.0f,
        tint
    );

    rlDrawRenderBatchActive();

    if (!customBlend)
        rlEnableColorBlend();
    else
        EndBlendMode();
}

ProcessShader* RenderGraph::FindShader(const std::string& name) {
    auto mapIt = shadersIndex_.find(name);
    if (mapIt != shadersIndex_.end() && mapIt->second < shaders_.size() &&
        shaders_[mapIt->second].name == name) {
        return &shaders_[mapIt->second];
    }
    // Defensive fallback -- should never trigger if the index is kept in
    // sync, but keeps this correct even if some future edit misses a spot.
    auto iterator = std::find_if(
        shaders_.begin(),
        shaders_.end(),
        [&](const ProcessShader& shader) {
            return shader.name == name;
        });

    return iterator == shaders_.end() ? nullptr : &(*iterator);
}

const ProcessShader* RenderGraph::FindShader(const std::string& name) const {
    auto mapIt = shadersIndex_.find(name);
    if (mapIt != shadersIndex_.end() && mapIt->second < shaders_.size() &&
        shaders_[mapIt->second].name == name) {
        return &shaders_[mapIt->second];
    }
    auto iterator = std::find_if(
        shaders_.begin(),
        shaders_.end(),
        [&](const ProcessShader& shader) {
            return shader.name == name;
        });

    return iterator == shaders_.end() ? nullptr : &(*iterator);
}

ProcessPass* RenderGraph::FindPass(const std::string& name) {
    auto mapIt = passesIndex_.find(name);
    if (mapIt != passesIndex_.end() && mapIt->second < passes_.size() &&
        passes_[mapIt->second].name == name) {
        return &passes_[mapIt->second];
    }
    auto iterator = std::find_if(
        passes_.begin(),
        passes_.end(),
        [&](const ProcessPass& pass) {
            return pass.name == name;
        }
    );

    return iterator == passes_.end() ? nullptr : &(*iterator);
}

const ProcessPass* RenderGraph::FindPass(const std::string& name) const {
    auto mapIt = passesIndex_.find(name);
    if (mapIt != passesIndex_.end() && mapIt->second < passes_.size() &&
        passes_[mapIt->second].name == name) {
        return &passes_[mapIt->second];
    }
    auto iterator = std::find_if(
        passes_.begin(),
        passes_.end(),
        [&](const ProcessPass& pass) {
            return pass.name == name;
        }
    );

    return iterator == passes_.end() ? nullptr : &(*iterator);
}

std::vector<ProcessPass>& RenderGraph::GetPasses() {
    return passes_;
}

const std::vector<ProcessPass>& RenderGraph::GetPasses() const {
    return passes_;
}

std::vector<ProcessGroup>& RenderGraph::GetGroups() {
    return groups_;
}

const std::vector<ProcessGroup>& RenderGraph::GetGroups() const {
    return groups_;
}

std::vector<ProcessShader>& RenderGraph::GetShaders() {
    return shaders_;
}

const std::vector<ProcessShader>& RenderGraph::GetShaders() const {
    return shaders_;
}

std::vector<ProcessGroupTemplate>& RenderGraph::GetGroupTemplates() {
    return groupTemplates_;
}

const std::vector<ProcessGroupTemplate>& RenderGraph::GetGroupTemplates() const {
    return groupTemplates_;
}

std::vector<ProcessGroupInstance>& RenderGraph::GetGroupInstances() {
    return groupInstances_;
}

const std::vector<ProcessGroupInstance>& RenderGraph::GetGroupInstances() const {
    return groupInstances_;
}

bool RenderGraph::MovePassUp(const std::string& passName) {
    ProcessPass* passPtr = FindPass(passName);
    if (!passPtr) return false;
    ProcessPass& pass = *passPtr;

    ProcessGroup* group =
        FindGroupContainingPass(pass.name);

    if (group) {
        auto iterator =
            std::find(group->passNames.begin(),
                      group->passNames.end(),
                      pass.name);

        if (iterator == group->passNames.begin())
            return false;

        std::iter_swap(iterator, iterator - 1);
        return true;
    }

    auto nodeIterator =
        std::find_if(renderOrder_.begin(), renderOrder_.end(),
                     [&](const RenderNode& node) {
                         return node.type == RenderNodeType::Pass &&
                                node.name == pass.name;
                     });

    if (nodeIterator == renderOrder_.begin() ||
        nodeIterator == renderOrder_.end()) {
        return false;
    }

    auto previous = nodeIterator - 1;

    while (true) {
        if (previous->type == RenderNodeType::Group ||
            !IsPassGrouped(previous->name)) {
            std::iter_swap(nodeIterator, previous);
            return true;
        }

        if (previous == renderOrder_.begin())
            break;

        --previous;
    }

    return false;
}

bool RenderGraph::MovePassDown(const std::string& passName) {
    ProcessPass* passPtr = FindPass(passName);
    if (!passPtr) return false;
    ProcessPass& pass = *passPtr;

    ProcessGroup* group =
        FindGroupContainingPass(pass.name);

    if (group) {
        auto iterator =
            std::find(group->passNames.begin(),
                      group->passNames.end(),
                      pass.name);

        if (iterator == group->passNames.end() ||
            iterator + 1 == group->passNames.end()) {
            return false;
        }

        std::iter_swap(iterator, iterator + 1);
        return true;
    }

    auto nodeIterator =
        std::find_if(renderOrder_.begin(), renderOrder_.end(),
                     [&](const RenderNode& node) {
                         return node.type == RenderNodeType::Pass &&
                                node.name == pass.name;
                     });

    if (nodeIterator == renderOrder_.end())
        return false;

    auto next = nodeIterator + 1;

    while (next != renderOrder_.end()) {
        if (next->type == RenderNodeType::Group ||
            !IsPassGrouped(next->name)) {
            std::iter_swap(nodeIterator, next);
            return true;
        }

        ++next;
    }

    return false;
}

void RenderGraph::SetPreviewAllPassesEnabled(bool enable) {
    if (previewAllPasses_ == enable) return;

    previewAllPasses_ = enable;

    if (enable) {
        EnsureAllPassOutputsAllocated();
    } else {
        for (ProcessPass& pass : passes_) {
            if (!pass.persistentOutput && pass.output.id != 0) {
                ReleaseToPool(pass.output, pass.outputFormat, pass.outputWithDepth);
                pass.output = {};
            }
        }
    }
}

bool RenderGraph::GetPreviewAllPasses() const {
    return previewAllPasses_;
}

const Texture2D* RenderGraph::GetPassOutputTexture(const std::string& passName) const {
    const ProcessPass* pass = FindPass(passName);

    if (!pass || pass->output.id == 0)
        return nullptr;

    return &pass->output.texture;
}

const Texture2D* RenderGraph::GetGroupOutputTexture(const std::string& groupName) const {
    const ProcessGroup* group = FindGroup(groupName);

    if (!group || group->output.id == 0)
        return nullptr;

    return &group->output.texture;
}

const Texture2D& RenderGraph::GetOutputTexture() const {
    return currentOutput_;
}

const RenderTexture2D& RenderGraph::GetSceneTexture() const {
    return sceneTexture_;
}

int RenderGraph::GetWidth() const {
    return width_;
}

int RenderGraph::GetHeight() const {
    return height_;
}

// ============================================================
// Compute / SSBO implementation
// ============================================================

// Shared tail of CreateComputeShader / CreateComputeShaderFromSource: link
// the already-compiled compute stage (csId) into a standalone program,
// register it under 'name', and re-point any existing passes' cached
// programDefinition (same shift hazard as elsewhere -- computePrograms_
// is a vector, so pushing back can reallocate and invalidate old pointers).
// sourcePath is stored as-is for record-keeping / ReloadComputeShader;
// pass "" for shaders that have no backing file (from-source shaders),
// which makes ReloadComputeShader's LoadFileText("") fail harmlessly.
ComputeShaderHandle RenderGraph::FinishComputeShaderCreation(const std::string& name,
                                                              const std::string& sourcePath,
                                                              unsigned int csId) {
    unsigned int programId = rlLoadShaderProgramCompute(csId);
    rlUnloadShader(csId);
    if (programId == 0) return ComputeShaderHandle();

    ComputeProgram program;
    program.name = name;
    program.sourcePath = sourcePath;
    program.id = programId;
    computePrograms_.push_back(std::move(program));
    computeProgramsIndex_[name] = computePrograms_.size() - 1;

    for (ComputePass& pass : computePasses_) {
        auto it = std::find_if(computePrograms_.begin(), computePrograms_.end(),
            [&](const ComputeProgram& candidate) { return candidate.name == pass.programName; });
        pass.programDefinition = it == computePrograms_.end() ? nullptr : &(*it);
    }
    return ComputeShaderHandle(name);
}

ComputeShaderHandle RenderGraph::CreateComputeShader(const std::string& name, const std::string& computeShaderPath) {
    if (name.empty() || computeShaderPath.empty() || FindComputeProgram(name))
        return ComputeShaderHandle();

    // No LoadShader equivalent exists for compute shaders -- this is raw
    // rlgl end to end. LoadFileText/UnloadFileText are raylib helpers for
    // reading a text file into a heap buffer; everything after that is
    // rlgl: compile the compute stage, link it alone (no vertex/fragment
    // pairing) into a standalone program.
    char* source = LoadFileText(computeShaderPath.c_str());
    if (source == nullptr) return ComputeShaderHandle();
    unsigned int csId = rlLoadShader(source, RL_COMPUTE_SHADER);
    UnloadFileText(source);

    return FinishComputeShaderCreation(name, computeShaderPath, csId);
}

ComputeShaderHandle RenderGraph::CreateComputeShaderFromSource(const std::string& name, const std::string& source) {
    if (name.empty() || source.empty() || FindComputeProgram(name))
        return ComputeShaderHandle();

    // Same rlgl compile/link path as CreateComputeShader, just skipping
    // the LoadFileText/UnloadFileText round trip since the caller already
    // has the GLSL text in memory (e.g. ShaderKernelLibrary's generated
    // per-kernel source).
    unsigned int csId = rlLoadShader(source.c_str(), RL_COMPUTE_SHADER);

    return FinishComputeShaderCreation(name, "", csId);
}

bool RenderGraph::DestroyComputeShader(const std::string& name) {
    auto it = std::find_if(computePrograms_.begin(), computePrograms_.end(),
        [&](const ComputeProgram& program) { return program.name == name; });
    if (it == computePrograms_.end()) return false;

    for (const ComputePass& pass : computePasses_)
        if (pass.programName == name) return false;

    if (it->id != 0) rlUnloadShaderProgram(it->id);
    computePrograms_.erase(it);
    RebuildNameIndex(computePrograms_, computeProgramsIndex_);

    // Same shift hazard as DestroyShader: any other pass's programDefinition
    // pointing past the erased slot is now stale. programName == name is
    // already blocked above, so this only re-resolves unrelated passes.
    for (ComputePass& pass : computePasses_) {
        auto progIt = std::find_if(computePrograms_.begin(), computePrograms_.end(),
            [&](const ComputeProgram& candidate) { return candidate.name == pass.programName; });
        pass.programDefinition = progIt == computePrograms_.end() ? nullptr : &(*progIt);
    }
    return true;
}

ComputeProgram* RenderGraph::FindComputeProgram(const std::string& name) {
    auto mapIt = computeProgramsIndex_.find(name);
    if (mapIt != computeProgramsIndex_.end() && mapIt->second < computePrograms_.size() &&
        computePrograms_[mapIt->second].name == name) {
        return &computePrograms_[mapIt->second];
    }
    auto it = std::find_if(computePrograms_.begin(), computePrograms_.end(),
        [&](const ComputeProgram& program) { return program.name == name; });
    return it == computePrograms_.end() ? nullptr : &(*it);
}

const ComputeProgram* RenderGraph::FindComputeProgram(const std::string& name) const {
    auto mapIt = computeProgramsIndex_.find(name);
    if (mapIt != computeProgramsIndex_.end() && mapIt->second < computePrograms_.size() &&
        computePrograms_[mapIt->second].name == name) {
        return &computePrograms_[mapIt->second];
    }
    auto it = std::find_if(computePrograms_.begin(), computePrograms_.end(),
        [&](const ComputeProgram& program) { return program.name == name; });
    return it == computePrograms_.end() ? nullptr : &(*it);
}

BufferHandle RenderGraph::CreateBuffer(const std::string& name, unsigned int sizeBytes,
                                          const void* initialData, int usageHint) {
    if (name.empty() || sizeBytes == 0 || FindBuffer(name))
        return BufferHandle();

    const unsigned int id = rlLoadShaderBuffer(sizeBytes, initialData, usageHint);
    if (id == 0) return BufferHandle();

    ShaderBuffer buffer;
    buffer.name = name;
    buffer.id = id;
    buffer.size = sizeBytes;
    buffer.usageHint = usageHint;
    buffers_.push_back(buffer);
    buffersIndex_[name] = buffers_.size() - 1;
    return BufferHandle(name);
}

bool RenderGraph::DestroyBuffer(const std::string& name) {
    auto it = std::find_if(buffers_.begin(), buffers_.end(),
        [&](const ShaderBuffer& buffer) { return buffer.name == name; });
    if (it == buffers_.end()) return false;

    // A buffer still bound in a compute pass would leave that pass
    // pointing at a dangling binding index with no backing buffer --
    // require it to be unbound first, same pattern DestroyPass/DestroyShader
    // already use for their own dependents.
    for (const ComputePass& pass : computePasses_)
        for (const ComputeBufferBinding& binding : pass.bufferBindings)
            if (binding.bufferName == name) return false;

    if (it->id != 0) rlUnloadShaderBuffer(it->id);
    buffers_.erase(it);
    RebuildNameIndex(buffers_, buffersIndex_);
    return true;
}

bool RenderGraph::ResizeBuffer(const std::string& name, unsigned int newSizeBytes, bool preserveData) {
    ShaderBuffer* buffer = FindBuffer(name);
    if (!buffer || buffer->id == 0 || newSizeBytes == 0) return false;
    if (buffer->size == newSizeBytes) return true;  // nothing to do

    // Read back the overlapping region before the old buffer is gone.
    std::vector<unsigned char> preserved;
    unsigned int copyBytes = 0;
    if (preserveData) {
        copyBytes = std::min(buffer->size, newSizeBytes);
        if (copyBytes > 0) {
            preserved.resize(copyBytes);
            rlReadShaderBuffer(buffer->id, preserved.data(), copyBytes, 0);
        }
    }

    const unsigned int newId = rlLoadShaderBuffer(newSizeBytes, nullptr, buffer->usageHint);
    if (newId == 0) return false;  // old buffer/id left untouched on failure

    if (copyBytes > 0) {
        rlUpdateShaderBuffer(newId, preserved.data(), copyBytes, 0);
    }

    rlUnloadShaderBuffer(buffer->id);
    buffer->id = newId;
    buffer->size = newSizeBytes;
    return true;
}

bool RenderGraph::UpdateBuffer(const std::string& name, const void* data, unsigned int sizeBytes, unsigned int offset) {
    ShaderBuffer* buffer = FindBuffer(name);
    if (!buffer || buffer->id == 0) return false;
    if (offset + sizeBytes > buffer->size) return false;

    rlUpdateShaderBuffer(buffer->id, data, sizeBytes, offset);
    return true;
}

bool RenderGraph::ReadBuffer(const std::string& name, void* outData, unsigned int sizeBytes, unsigned int offset) {
    ShaderBuffer* buffer = FindBuffer(name);
    if (!buffer || buffer->id == 0) return false;
    if (offset + sizeBytes > buffer->size) return false;

    rlReadShaderBuffer(buffer->id, outData, sizeBytes, offset);
    return true;
}

ShaderBuffer* RenderGraph::FindBuffer(const std::string& name) {
    auto mapIt = buffersIndex_.find(name);
    if (mapIt != buffersIndex_.end() && mapIt->second < buffers_.size() &&
        buffers_[mapIt->second].name == name) {
        return &buffers_[mapIt->second];
    }
    auto it = std::find_if(buffers_.begin(), buffers_.end(),
        [&](const ShaderBuffer& buffer) { return buffer.name == name; });
    return it == buffers_.end() ? nullptr : &(*it);
}

const ShaderBuffer* RenderGraph::FindBuffer(const std::string& name) const {
    auto mapIt = buffersIndex_.find(name);
    if (mapIt != buffersIndex_.end() && mapIt->second < buffers_.size() &&
        buffers_[mapIt->second].name == name) {
        return &buffers_[mapIt->second];
    }
    auto it = std::find_if(buffers_.begin(), buffers_.end(),
        [&](const ShaderBuffer& buffer) { return buffer.name == name; });
    return it == buffers_.end() ? nullptr : &(*it);
}

ComputePassHandle RenderGraph::CreateComputePass(const std::string& name, const std::string& computeShaderName) {
    if (name.empty() || FindComputePass(name) || FindComputeGroup(name))
        return ComputePassHandle();

    ComputeProgram* program = FindComputeProgram(computeShaderName);
    if (!program) return ComputePassHandle();

    ComputePass pass;
    pass.name = name;
    pass.programName = computeShaderName;
    pass.programDefinition = program;
    computePasses_.push_back(std::move(pass));
    computePassesIndex_[name] = computePasses_.size() - 1;
    return ComputePassHandle(name);
}

bool RenderGraph::DestroyComputePass(const std::string& name) {
    auto it = std::find_if(computePasses_.begin(), computePasses_.end(),
        [&](const ComputePass& pass) { return pass.name == name; });
    if (it == computePasses_.end()) return false;
    //Destroy from any group if present
    for (ComputeGroup& group : computeGroups_) {
        group.computePassNames.erase(
            std::remove(group.computePassNames.begin(), group.computePassNames.end(), name),
            group.computePassNames.end()
        );
    }

    // Remove from render graph if present
    renderOrder_.erase(
        std::remove_if(renderOrder_.begin(), renderOrder_.end(),
            [&](const RenderNode& node) {
                return node.type == RenderNodeType::Compute && node.name == name;
            }),
        renderOrder_.end()
    );

    computePasses_.erase(it);
    RebuildNameIndex(computePasses_, computePassesIndex_);
    return true;
}

bool RenderGraph::BindBufferToComputePass(const std::string& passName, const std::string& bufferName, unsigned int bindingIndex) {
    ComputePass* pass = FindComputePass(passName);
    ShaderBuffer* buffer = FindBuffer(bufferName);
    if (!pass || !buffer) return false;

    auto it = std::find_if(pass->bufferBindings.begin(), pass->bufferBindings.end(),
        [&](const ComputeBufferBinding& binding) { return binding.bindingIndex == bindingIndex; });

    if (it != pass->bufferBindings.end()) {
        it->bufferName = bufferName;
        return true;
    }

    pass->bufferBindings.push_back({bufferName, bindingIndex});
    return true;
}

bool RenderGraph::BindBufferSlotByRegistryName(const std::string& passName, const std::string& blockName, unsigned int bindingIndex) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass || blockName.empty()) return false;

    auto it = std::find_if(pass->bufferBindings.begin(), pass->bufferBindings.end(),
        [&](const ComputeBufferBinding& binding) { return binding.bindingIndex == bindingIndex; });

    if (it != pass->bufferBindings.end()) {
        it->bufferName = blockName;
        return true;
    }

    pass->bufferBindings.push_back({blockName, bindingIndex});
    return true;
}
bool RenderGraph::SwapComputePassBufferBindings(const std::string& passName,
                                                unsigned int bindingIndexA,
                                                unsigned int bindingIndexB) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;

    auto itA = std::find_if(pass->bufferBindings.begin(), pass->bufferBindings.end(),
                            [&](const ComputeBufferBinding& b) { return b.bindingIndex == bindingIndexA; });
    auto itB = std::find_if(pass->bufferBindings.begin(), pass->bufferBindings.end(),
                            [&](const ComputeBufferBinding& b) { return b.bindingIndex == bindingIndexB; });

    if (itA == pass->bufferBindings.end() || itB == pass->bufferBindings.end())
        return false;

    std::swap(itA->bufferName, itB->bufferName);
    return true;
}
bool RenderGraph::UnbindBufferFromComputePass(const std::string& passName, unsigned int bindingIndex) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;

    auto it = std::find_if(pass->bufferBindings.begin(), pass->bufferBindings.end(),
        [&](const ComputeBufferBinding& binding) { return binding.bindingIndex == bindingIndex; });
    if (it == pass->bufferBindings.end()) return false;

    pass->bufferBindings.erase(it);
    return true;
}
bool RenderGraph::BindBufferToGeometryPass(const std::string& passName,
                                           const std::string& bufferName,
                                           unsigned int bindingIndex) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;
    if (!FindBuffer(bufferName)) return false;

    // Replace existing binding at the same index
    pass->bufferBindings.erase(
        std::remove_if(pass->bufferBindings.begin(),
                       pass->bufferBindings.end(),
                       [&](const ComputeBufferBinding& b) {
                           return b.bindingIndex == bindingIndex;
                       }),
        pass->bufferBindings.end());

    pass->bufferBindings.push_back({bufferName, bindingIndex});
    return true;
}
ComputeGroupHandle RenderGraph::CreateComputeGroup(const std::string& name) {
    if (name.empty() || FindComputeGroup(name) || FindPass(name) ||
        FindGroup(name) || FindComputePass(name) || FindGeometryPass(name)) {
        SetError("AddComputeGroup: invalid or duplicate name '" + name + "'");
        return ComputeGroupHandle();
    }

    ComputeGroup group;
    group.name = name;
    computeGroups_.push_back(std::move(group));
    return ComputeGroupHandle(name);
}

bool RenderGraph::DestroyComputeGroup(const std::string& name) {
    auto it = std::find_if(computeGroups_.begin(), computeGroups_.end(),
        [&](const ComputeGroup& g) { return g.name == name; });
    if (it == computeGroups_.end()) return false;

    // Remove from render order if present
    renderOrder_.erase(
        std::remove_if(renderOrder_.begin(), renderOrder_.end(),
            [&](const RenderNode& node) {
                return node.type == RenderNodeType::ComputeGroup && node.name == name;
            }),
        renderOrder_.end()
    );

    computeGroups_.erase(it);
    return true;
}

bool RenderGraph::AttachComputePassToGroup(const std::string& groupName, const std::string& passName) {
    ComputeGroup* group = FindComputeGroup(groupName);
    ComputePass* pass = FindComputePass(passName);
    if (!group || !pass) return false;

    // Check if pass already in another group
    for (const auto& g : computeGroups_) {
        if (std::find(g.computePassNames.begin(), g.computePassNames.end(), passName) != g.computePassNames.end())
            return false;
    }

    group->computePassNames.push_back(passName);
    return true;
}

bool RenderGraph::RemoveComputePassFromGroup(const std::string& groupName, const std::string& passName) {
    ComputeGroup* group = FindComputeGroup(groupName);
    if (!group) return false;

    auto it = std::find(group->computePassNames.begin(), group->computePassNames.end(), passName);
    if (it == group->computePassNames.end()) return false;

    group->computePassNames.erase(it);
    return true;
}

bool RenderGraph::SetComputeGroupEnabled(const std::string& groupName, bool enabled) {
    ComputeGroup* group = FindComputeGroup(groupName);
    if (!group) return false;
    group->enabled = enabled;
    return true;
}

bool RenderGraph::SetComputeGroupCondition(const std::string& groupName, std::function<bool()> condition) {
    ComputeGroup* group = FindComputeGroup(groupName);
    if (!group) return false;
    group->condition = std::move(condition);
    return true;
}

bool RenderGraph::SetComputeGroupPriority(const std::string& groupName, int priority) {
    ComputeGroup* group = FindComputeGroup(groupName);
    if (!group) return false;
    group->priority = priority;
    return true;
}

ComputeGroup* RenderGraph::FindComputeGroup(const std::string& name) {
    auto it = std::find_if(computeGroups_.begin(), computeGroups_.end(),
        [&](const ComputeGroup& g) { return g.name == name; });
    return it == computeGroups_.end() ? nullptr : &(*it);
}

const ComputeGroup* RenderGraph::FindComputeGroup(const std::string& name) const {
    auto it = std::find_if(computeGroups_.begin(), computeGroups_.end(),
        [&](const ComputeGroup& g) { return g.name == name; });
    return it == computeGroups_.end() ? nullptr : &(*it);
}

std::vector<ComputeGroup>& RenderGraph::GetComputeGroups() {
    return computeGroups_;
}

const std::vector<ComputeGroup>& RenderGraph::GetComputeGroups() const {
    return computeGroups_;
}
bool RenderGraph::UnbindBufferFromGeometryPass(const std::string& passName,
                                               unsigned int bindingIndex) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;

    auto it = std::find_if(pass->bufferBindings.begin(),
                           pass->bufferBindings.end(),
                           [&](const ComputeBufferBinding& b) {
                               return b.bindingIndex == bindingIndex;
                           });
    if (it == pass->bufferBindings.end()) return false;

    pass->bufferBindings.erase(it);
    return true;
}
bool RenderGraph::SwapGeometryPassBufferBindings(const std::string& passName,
                                                 unsigned int bindingIndexA,
                                                 unsigned int bindingIndexB) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;

    auto itA = std::find_if(pass->bufferBindings.begin(), pass->bufferBindings.end(),
                            [&](const ComputeBufferBinding& b) { return b.bindingIndex == bindingIndexA; });
    auto itB = std::find_if(pass->bufferBindings.begin(), pass->bufferBindings.end(),
                            [&](const ComputeBufferBinding& b) { return b.bindingIndex == bindingIndexB; });

    if (itA == pass->bufferBindings.end() || itB == pass->bufferBindings.end())
        return false;

    std::swap(itA->bufferName, itB->bufferName);
    return true;
}
bool RenderGraph::AttachComputePassToGraph(const std::string& passName) {
    if (!FindComputePass(passName)) return false;

    // Check if already in graph
    auto it = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::Compute && node.name == passName;
        });
    if (it != renderOrder_.end()) return false;

    renderOrder_.push_back({RenderNodeType::Compute, passName});
    return true;
}

bool RenderGraph::RemoveComputePassFromGraph(const std::string& passName) {
    auto it = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::Compute && node.name == passName;
        });
    if (it == renderOrder_.end()) return false;

    renderOrder_.erase(it);
    return true;
}

bool RenderGraph::InsertComputePassBefore(const std::string& computePassName,
                                                 const std::string& beforeNodeName) {
    if (!FindComputePass(computePassName)) return false;

    auto beforeIt = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) { return node.name == beforeNodeName; });
    if (beforeIt == renderOrder_.end()) return false;

    // Ensure compute pass isn't already in graph
    auto computeIt = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::Compute && node.name == computePassName;
        });
    if (computeIt != renderOrder_.end()) return false;

    renderOrder_.insert(beforeIt, {RenderNodeType::Compute, computePassName});
    return true;
}

bool RenderGraph::InsertComputePassAfter(const std::string& computePassName,
                                                const std::string& afterNodeName) {
    if (!FindComputePass(computePassName)) return false;

    auto afterIt = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) { return node.name == afterNodeName; });
    if (afterIt == renderOrder_.end()) return false;

    auto computeIt = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::Compute && node.name == computePassName;
        });
    if (computeIt != renderOrder_.end()) return false;

    renderOrder_.insert(afterIt + 1, {RenderNodeType::Compute, computePassName});
    return true;
}
bool RenderGraph::SetComputeWorkgroupSize(const std::string& passName, unsigned int x, unsigned int y, unsigned int z) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass || x == 0 || y == 0 || z == 0) return false;

    pass->groupsX = x;
    pass->groupsY = y;
    pass->groupsZ = z;
    return true;
}

// ------------------------------------------------------------------
// Kernel shaders (HLSL-style multi-kernel, or plain GLSL -- see
// ShaderKernelLibrary in render_graph.h) and automatic uniforms.
// ------------------------------------------------------------------

void RenderGraph::LoadKernelFile(const std::string& path, const std::string& plainKernelName) {
    kernelLibrary_.LoadFile(path, plainKernelName);
}

void RenderGraph::LoadKernelSource(const std::string& source, const std::string& plainKernelName) {
    kernelLibrary_.LoadSource(source, plainKernelName);
}

const KernelInfo* RenderGraph::FindKernel(const std::string& name) const {
    if (!kernelLibrary_.HasKernel(name)) return nullptr;
    return &kernelLibrary_.GetKernel(name);
}

std::vector<std::string> RenderGraph::KernelNames() const {
    return kernelLibrary_.KernelNames();
}

void RenderGraph::RegisterImage(const std::string& name, Texture2D texture) {
    kernelImageRegistry_[name] = texture;
}

void RenderGraph::RegisterSampler(const std::string& name, Texture2D texture) {
    kernelSamplerRegistry_[name] = texture;
}

void RenderGraph::RegisterTexture(const std::string& name, Texture2D texture) {
    RegisterImage(name, texture);
    RegisterSampler(name, texture);
}

void RenderGraph::RegisterBuffer(const std::string& name, const BufferHandle& buffer) {
    kernelBufferRegistry_[name] = buffer;
}

ComputePassHandle RenderGraph::CreateComputePassFromKernel(const std::string& kernelName,
                                                             unsigned int itemsX,
                                                             unsigned int itemsY,
                                                             unsigned int itemsZ) {
    if (!kernelLibrary_.HasKernel(kernelName))
        throw std::runtime_error("CreateComputePassFromKernel: no kernel named '" + kernelName + "' (call LoadKernelFile/LoadKernelSource first)");
    const KernelInfo& kernel = kernelLibrary_.GetKernel(kernelName);

    ComputeShaderHandle shader = CreateComputeShaderFromSource(kernel.name, kernel.fullSource);
    RequireKernelBindingStep(shader.IsValid(), "compile/register shader for kernel '" + kernel.name + "' (duplicate name, or GLSL compile error)");

    ComputePassHandle pass = CreateComputePass(kernel.name, kernel.name);
    RequireKernelBindingStep(pass.IsValid(), "create pass for kernel '" + kernel.name + "'");

    for (auto& img : kernel.images) {
        // Bound at the exact binding number the shader declares
        // (layout(..., binding = N)), or the next free unit if the shader
        // left binding unspecified -- either way it's already resolved by
        // ShaderKernelLibrary's parse-time auto-assignment, nothing to
        // specify by hand here.
        ComputeImageBindingHandle binding = CreateComputeImageBinding(
            pass, img.name + "Binding", img.name, false, ToRLFormat(img.format), img.binding);
        RequireKernelBindingStep(binding.IsValid(), "bind image '" + img.name + "' on kernel '" + kernel.name + "' (uniform not found in compiled shader, or duplicate binding)");
        // Wired to the image registry by name, not resolved now -- see
        // SetComputeImageRegistrySource. RegisterImage(img.name, ...) can
        // be called before or after this; the lookup happens fresh every
        // dispatch, so registration order never matters. Run Validate()
        // after setup to catch a name that never gets registered at all.
        SetComputeImageRegistrySource(binding, img.name);
    }

    for (auto& tex : kernel.textures) {
        ComputeTextureBindingHandle binding = CreateComputeTextureBinding(pass, tex.name + "Binding", tex.name);
        RequireKernelBindingStep(binding.IsValid(), "bind texture '" + tex.name + "' on kernel '" + kernel.name + "' (uniform not found in compiled shader, or duplicate binding)");
        // Same deal, against the sampler registry -- see RegisterSampler.
        SetComputeTextureRegistrySource(binding, tex.name);
    }

    for (auto& buf : kernel.buffers) {
        // Bound by the kernel's block name rather than a concrete buffer
        // name -- DispatchComputePass's SSBO bind loop resolves it
        // through kernelBufferRegistry_ every dispatch (falling back to a
        // literal buffer named buf.blockName first, same as
        // BindBufferToComputePass), so RegisterBuffer(buf.blockName, ...)
        // can likewise be called before or after this.
        RequireKernelBindingStep(BindBufferSlotByRegistryName(kernel.name, buf.blockName, buf.binding),
                     "bind buffer '" + buf.blockName + "' on kernel '" + kernel.name + "'");
    }

    unsigned int gx, gy, gz;
    kernel.ComputeGroupCounts(itemsX, itemsY, itemsZ, gx, gy, gz);
    RequireKernelBindingStep(SetComputeWorkgroupSize(kernel.name, gx, gy, gz), "set workgroup size for kernel '" + kernel.name + "'");

    RequireKernelBindingStep(AttachComputePassToGraph(kernel.name), "attach kernel '" + kernel.name + "' to the graph");

    // Remember which kernel this pass came from so Apply() can auto-push
    // global uniforms (see SetUniform/ApplyGlobalUniformsToKernelPasses)
    // to exactly the passes that declare each name.
    computePassKernelName_[pass.name] = kernel.name;

    return pass;
}

void RenderGraph::SetUniform(const std::string& name, float value) { StoreGlobalUniform(name, value); }
void RenderGraph::SetUniform(const std::string& name, int value) { StoreGlobalUniform(name, value); }
void RenderGraph::SetUniform(const std::string& name, Vector2 value) { StoreGlobalUniform(name, value); }
void RenderGraph::SetUniform(const std::string& name, Vector3 value) { StoreGlobalUniform(name, value); }
void RenderGraph::SetUniform(const std::string& name, Vector4 value) { StoreGlobalUniform(name, value); }
void RenderGraph::SetUniform(const std::string& name, Color value) { StoreGlobalUniform(name, value); }

void RenderGraph::ApplyGlobalUniformsToKernelPasses() {
    for (auto& [passName, kernelName] : computePassKernelName_) {
        if (!kernelLibrary_.HasKernel(kernelName)) continue;
        const KernelInfo& kernel = kernelLibrary_.GetKernel(kernelName);
        auto& lastApplied = globalUniformLastAppliedToPass_[passName];

        for (const auto& [uniformName, entry] : globalUniforms_) {
            if (!kernel.HasUniform(uniformName)) continue;
            uint64_t& last = lastApplied[uniformName]; // 0 if never applied to this pass before
            if (entry.version <= last) continue;

            std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, float>) SetComputeFloat(passName, uniformName, v);
                else if constexpr (std::is_same_v<T, int>) SetComputeInt(passName, uniformName, v);
                else if constexpr (std::is_same_v<T, Vector2>) SetComputeVector2(passName, uniformName, v);
                else if constexpr (std::is_same_v<T, Vector3>) SetComputeVector3(passName, uniformName, v);
                else if constexpr (std::is_same_v<T, Vector4>) SetComputeVector4(passName, uniformName, v);
                else if constexpr (std::is_same_v<T, Color>) SetComputeColor(passName, uniformName, v);
                // std::monostate: never actually stored via StoreGlobalUniform, unreachable.
            }, entry.value);

            last = entry.version;
        }
    }
}

bool RenderGraph::AttachComputeGroupToGraph(const std::string& groupName) {
    if (!FindComputeGroup(groupName)) return false;

    auto it = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::ComputeGroup && node.name == groupName;
        });
    if (it != renderOrder_.end()) return false;

    renderOrder_.push_back({RenderNodeType::ComputeGroup, groupName});
    return true;
}

bool RenderGraph::RemoveComputeGroupFromGraph(const std::string& groupName) {
    auto it = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::ComputeGroup && node.name == groupName;
        });
    if (it == renderOrder_.end()) return false;

    renderOrder_.erase(it);
    return true;
}

bool RenderGraph::InsertComputeGroupBefore(const std::string& computeGroupName,
                                           const std::string& beforeNodeName) {
    if (!FindComputeGroup(computeGroupName)) return false;

    auto beforeIt = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) { return node.name == beforeNodeName; });
    if (beforeIt == renderOrder_.end()) return false;

    // Check if already present
    auto already = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::ComputeGroup && node.name == computeGroupName;
        });
    if (already != renderOrder_.end()) return false;

    renderOrder_.insert(beforeIt, {RenderNodeType::ComputeGroup, computeGroupName});
    return true;
}

bool RenderGraph::InsertComputeGroupAfter(const std::string& computeGroupName,
                                          const std::string& afterNodeName) {
    if (!FindComputeGroup(computeGroupName)) return false;

    auto afterIt = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) { return node.name == afterNodeName; });
    if (afterIt == renderOrder_.end()) return false;

    auto already = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::ComputeGroup && node.name == computeGroupName;
        });
    if (already != renderOrder_.end()) return false;

    renderOrder_.insert(afterIt + 1, {RenderNodeType::ComputeGroup, computeGroupName});
    return true;
}
bool RenderGraph::SetComputePassEnabled(const std::string& passName, bool enabled) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;

    pass->enabled = enabled;
    return true;
}

bool RenderGraph::DispatchComputePass(const std::string& passName) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass || !pass->enabled || !pass->programDefinition || pass->programDefinition->id == 0)
        return false;

    if (pass->persistentState && currentShaderId_ == pass->programDefinition->id) {
        // Already bound by a previous persistent-state dispatch (this
        // frame or before) and nothing has re-enabled a different shader
        // since -- skip the redundant enable. Bindings are still refreshed
        // below regardless, since buffer/texture/image contents may have
        // changed even if the program hasn't.
    } else {
        rlEnableShader(pass->programDefinition->id);
        currentShaderId_ = pass->programDefinition->id;
    }
    ReapplyComputeUniforms(*pass);

    // Bind UBOs
    for (const auto& binding : pass->uniformBufferBindings) {
        UniformBuffer* buffer = FindUniformBuffer(binding.bufferName);
        if (buffer && buffer->id != 0) {
            glBindBufferBase(GL_UNIFORM_BUFFER, binding.bindingIndex, buffer->id);
        }
    }

    // Bind SSBOs
    for (const ComputeBufferBinding& binding : pass->bufferBindings) {
        const ShaderBuffer* buffer = FindBuffer(binding.bufferName);
        if (!buffer) {
            // binding.bufferName may be a kernel block name (from
            // CreateComputePassFromKernel) rather than a literal buffer
            // name -- fall back through the kernel buffer registry so
            // RegisterBuffer() can be called before or after the pass
            // was created and still resolve here, every dispatch.
            auto regIt = kernelBufferRegistry_.find(binding.bufferName);
            if (regIt != kernelBufferRegistry_.end())
                buffer = FindBuffer(regIt->second.name);
        }
        if (buffer && buffer->id != 0)
            rlBindShaderBuffer(buffer->id, binding.bindingIndex);
    }

    // Flush any pending compute-written textures before binding as samplers
    // We do this in the loop as we resolve the final texture ID.
    int textureUnit = 1; // leave unit 0 free
    for (const TextureBinding& binding : pass->textureBindings) {
        Texture2D texture = binding.texture;

        if (!binding.kernelRegistryName.empty()) {
            // Wired by CreateComputePassFromKernel -- resolved against
            // the live registry every dispatch, so it doesn't matter
            // whether RegisterSampler() ran before or after pass setup,
            // and a later re-register takes effect immediately.
            auto regIt = kernelSamplerRegistry_.find(binding.kernelRegistryName);
            if (regIt != kernelSamplerRegistry_.end()) texture = regIt->second;
        } else if (!binding.sourcePassName.empty()) {
            if (binding.sourcePassName == SceneSourceName()) {
                texture = sceneTexture_.texture;
            } else {
                const ProcessPass* srcPass = FindPass(binding.sourcePassName);
                if (srcPass && srcPass->output.id != 0)
                    texture = srcPass->output.texture;
                else {
                    const ProcessGroup* srcGroup = FindGroup(binding.sourcePassName);
                    if (srcGroup && srcGroup->output.id != 0)
                        texture = srcGroup->output.texture;
                    else {
                        Texture2D fbTexture = ResolveFramebufferColorTexture(binding.sourcePassName, binding.framebufferAttachmentIndex);
                        if (fbTexture.id != 0) texture = fbTexture;
                    }
                }
            }
        }

        if (texture.id != 0) {
            // Ensure any compute writes to this texture are visible to the sampler
            EnsureBarrierForTexture(texture.id);

            rlActiveTextureSlot(textureUnit);
            if (binding.isCubemap) {
                // rlEnableTexture always binds GL_TEXTURE_2D -- fine for
                // every other TextureBinding, wrong for a samplerCube.
                glBindTexture(GL_TEXTURE_CUBE_MAP, texture.id);
            } else if (binding.isTextureArray) {
                glBindTexture(GL_TEXTURE_2D_ARRAY, texture.id);
            } else if (binding.isTexture3D) {
                glBindTexture(GL_TEXTURE_3D, texture.id);
            } else {
                rlEnableTexture(texture.id);
            }
            int unit = textureUnit;
            rlSetUniform(binding.location, &unit, RL_SHADER_UNIFORM_INT, 1);
            textureUnit++;
        }
    }

    // Bind images (for read/write operations in compute)
    for (const ComputeImageBinding& binding : pass->imageBindings) {
        Texture2D texture = binding.texture;

        if (!binding.kernelRegistryName.empty()) {
            // Same live, order-independent resolution as the sampler
            // loop above, against the image registry instead.
            auto regIt = kernelImageRegistry_.find(binding.kernelRegistryName);
            if (regIt != kernelImageRegistry_.end()) texture = regIt->second;
        } else if (!binding.isRawTexture && !binding.sourcePassName.empty()) {
            if (binding.sourcePassName == SceneSourceName()) {
                texture = sceneTexture_.texture;
            } else {
                const ProcessPass* srcPass = FindPass(binding.sourcePassName);
                if (srcPass && srcPass->output.id != 0)
                    texture = srcPass->output.texture;
                else {
                    const ProcessGroup* srcGroup = FindGroup(binding.sourcePassName);
                    if (srcGroup && srcGroup->output.id != 0)
                        texture = srcGroup->output.texture;
                    else {
                        Texture2D fbTexture = ResolveFramebufferColorTexture(binding.sourcePassName);
                        if (fbTexture.id != 0) texture = fbTexture;
                    }
                }
            }
        }

        if (texture.id != 0) {
            // Determine the OpenGL internal format and access type
            unsigned int glInternalFormat = 0;
            GLenum access = binding.readOnly ? GL_READ_ONLY : GL_WRITE_ONLY;

            if (binding.isRawTexture) {
                glInternalFormat = binding.glInternalFormat;
            } else {
                // Convert raylib pixel format to OpenGL internal format
                unsigned int glFormat, glType;
                rlGetGlTextureFormats(binding.format, &glInternalFormat, &glFormat, &glType);
            }

            // If this is a read-only image, ensure previous compute writes are visible
            if (binding.readOnly) {
                EnsureBarrierForTexture(texture.id);
            }

            // Bind using raw OpenGL (works for any texture target)
            glBindImageTexture(binding.imageUnit, texture.id, 0, GL_FALSE, 0, access, glInternalFormat);

            // If this is a write access, mark the texture as written by this compute pass
            if (!binding.readOnly) {
                computeWrittenTextures_.insert(texture.id);
            }

            // Set the uniform to the image unit index
            int unit = static_cast<int>(binding.imageUnit);
            rlSetUniform(binding.location, &unit, RL_SHADER_UNIFORM_INT, 1);
        }
    }

    // Dispatch the compute shader
    rlComputeShaderDispatch(pass->groupsX, pass->groupsY, pass->groupsZ);

    // If this pass has any SSBO bindings, insert a conservative memory
    // barrier so writes are visible to whatever reads the buffer next
    // (another compute pass or a graphics pass). RenderGraph doesn't track
    // per-buffer read/write access modes, so this fires whenever a buffer is
    // bound at all rather than only on writes; the cost of an extra barrier
    // is negligible next to the cost of a silent data race.
    if (!pass->bufferBindings.empty()) {
        EmitMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // Unbind images (clean up)
    for (const ComputeImageBinding& binding : pass->imageBindings) {
        glBindImageTexture(binding.imageUnit, 0, 0, GL_FALSE, 0, GL_READ_WRITE, 0);
    }

    if (!pass->persistentState) {
        rlDisableShader();
        currentShaderId_ = 0;
    }
    // else: leave the program bound. Every other shader-enable call site in
    // this file (ExecutePass, geometry passes, direct uniform setters)
    // calls rlDisableShader() unconditionally when it's done and resets
    // currentShaderId_ to 0, so the next thing that actually needs a
    // shader always gets a correct, explicit rlEnableShader() -- this
    // cached id is only ever used to skip a *redundant* re-enable of the
    // exact same program back-to-back, never to skip binding a new one.
    return true;
}
ComputePass* RenderGraph::FindComputePass(const std::string& name) {
    auto mapIt = computePassesIndex_.find(name);
    if (mapIt != computePassesIndex_.end() && mapIt->second < computePasses_.size() &&
        computePasses_[mapIt->second].name == name) {
        return &computePasses_[mapIt->second];
    }
    auto it = std::find_if(computePasses_.begin(), computePasses_.end(),
        [&](const ComputePass& pass) { return pass.name == name; });
    return it == computePasses_.end() ? nullptr : &(*it);
}

const ComputePass* RenderGraph::FindComputePass(const std::string& name) const {
    auto mapIt = computePassesIndex_.find(name);
    if (mapIt != computePassesIndex_.end() && mapIt->second < computePasses_.size() &&
        computePasses_[mapIt->second].name == name) {
        return &computePasses_[mapIt->second];
    }
    auto it = std::find_if(computePasses_.begin(), computePasses_.end(),
        [&](const ComputePass& pass) { return pass.name == name; });
    return it == computePasses_.end() ? nullptr : &(*it);
}

std::vector<ComputeProgram>& RenderGraph::GetComputePrograms() {
    return computePrograms_;
}

std::vector<ComputePass>& RenderGraph::GetComputePasses() {
    return computePasses_;
}

std::vector<ShaderBuffer>& RenderGraph::GetBuffers() {
    return buffers_;
}
const std::vector<ComputeProgram>& RenderGraph::GetComputePrograms() const {
    return computePrograms_;
}

const std::vector<ComputePass>& RenderGraph::GetComputePasses() const {
    return computePasses_;
}

const std::vector<ShaderBuffer>& RenderGraph::GetBuffers() const {
    return buffers_;
}
// ============================================================
// Framebuffer management
// ============================================================

FramebufferHandle RenderGraph::CreateFramebuffer(const std::string& name,
                                         int width,
                                         int height,
                                         const std::vector<int>& colorFormats,
                                         bool withDepth,
                                         bool depthAsTexture) {
    if (name.empty() || FindFramebuffer(name) || width <= 0 || height <= 0 || colorFormats.empty())
        return FramebufferHandle();

    Framebuffer fb;
    fb.name = name;
    fb.width = width;
    fb.height = height;

    fb.id = rlLoadFramebuffer();
    if (fb.id == 0) return FramebufferHandle();

    // Bind the new framebuffer while we attach textures.
    rlEnableFramebuffer(fb.id);

    // Activate as many draw buffers as we have color attachments.
    rlActiveDrawBuffers(static_cast<int>(colorFormats.size()));

    for (size_t i = 0; i < colorFormats.size(); ++i) {
        unsigned int texId = rlLoadTexture(nullptr, width, height, colorFormats[i], 1);
        if (texId == 0) {
            // Cleanup on failure.
            for (auto& att : fb.colorAttachments) rlUnloadTexture(att.textureId);
            if (fb.depthRenderbufferId) rlUnloadTexture(fb.depthRenderbufferId);
            if (fb.depthTextureId) rlUnloadTexture(fb.depthTextureId);
            rlUnloadFramebuffer(fb.id);
            rlDisableFramebuffer();
            return FramebufferHandle();
        }

        int attachType = RL_ATTACHMENT_COLOR_CHANNEL0 + static_cast<int>(i);
        rlFramebufferAttach(fb.id, texId, attachType, RL_ATTACHMENT_TEXTURE2D, 0);

        FramebufferAttachment attachment;
        attachment.textureId = texId;
        attachment.format = colorFormats[i];
        attachment.attachType = attachType;
        fb.colorAttachments.push_back(attachment);
    }

    // Optionally attach depth, either as a renderbuffer (default -- usable
    // for depth testing only) or a texture (also readable afterward, e.g.
    // to reconstruct scene depth in a later pass).
    if (withDepth) {
        if (depthAsTexture) {
            fb.depthTextureId = rlLoadTextureDepth(width, height, false);
            if (fb.depthTextureId == 0) {
                for (auto& att : fb.colorAttachments) rlUnloadTexture(att.textureId);
                rlUnloadFramebuffer(fb.id);
                rlDisableFramebuffer();
                return FramebufferHandle();
            }
            rlFramebufferAttach(fb.id, fb.depthTextureId,
                                RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0);
        } else {
            fb.depthRenderbufferId = rlLoadTextureDepth(width, height, true);
            if (fb.depthRenderbufferId == 0) {
                // Cleanup on failure.
                for (auto& att : fb.colorAttachments) rlUnloadTexture(att.textureId);
                rlUnloadFramebuffer(fb.id);
                rlDisableFramebuffer();
                return FramebufferHandle();
            }
            rlFramebufferAttach(fb.id, fb.depthRenderbufferId,
                                RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_RENDERBUFFER, 0);
        }
    }

    // Verify completeness; this also unbinds the framebuffer.
    if (!rlFramebufferComplete(fb.id)) {
        for (auto& att : fb.colorAttachments) rlUnloadTexture(att.textureId);
        if (fb.depthRenderbufferId) rlUnloadTexture(fb.depthRenderbufferId);
        if (fb.depthTextureId) rlUnloadTexture(fb.depthTextureId);
        rlUnloadFramebuffer(fb.id);
        return FramebufferHandle();
    }

    // rlFramebufferComplete already unbinds; no need to call rlDisableFramebuffer.

    const std::string fbName = fb.name;
    framebuffers_.push_back(std::move(fb));
    framebuffersIndex_[fbName] = framebuffers_.size() - 1;
    return FramebufferHandle(fbName);
}

FramebufferHandle RenderGraph::CreateCubemapFramebuffer(const std::string& name,
                                         int faceSize,
                                         const std::vector<int>& colorFormats,
                                         bool withDepth) {
    if (name.empty() || FindFramebuffer(name) || faceSize <= 0 || colorFormats.empty())
        return FramebufferHandle();

    Framebuffer fb;
    fb.name = name;
    fb.width = faceSize;
    fb.height = faceSize;
    fb.isCubemap = true;
    fb.faceSize = faceSize;

    fb.id = rlLoadFramebuffer();
    if (fb.id == 0) return FramebufferHandle();

    rlEnableFramebuffer(fb.id);
    rlActiveDrawBuffers(static_cast<int>(colorFormats.size()));

    // Allocate one full cubemap object (all 6 faces) per color channel --
    // rlLoadTextureCubemap(nullptr, ...) allocates storage without
    // uploading data, same call raylib's own runtime skybox baking uses.
    // We attach face 0 here just to get a complete FBO for the validity
    // check below; CaptureFace-equivalent code (see the Geometry branch
    // in Apply()) re-points each attachment to the correct face with
    // rlFramebufferAttach right before every one of the 6 draws, so which
    // face is bound here doesn't matter.
    for (size_t i = 0; i < colorFormats.size(); ++i) {
        unsigned int texId = rlLoadTextureCubemap(nullptr, faceSize, colorFormats[i], 1);
        if (texId == 0) {
            for (auto& att : fb.colorAttachments) rlUnloadTexture(att.textureId);
            if (fb.depthRenderbufferId) rlUnloadTexture(fb.depthRenderbufferId);
            rlUnloadFramebuffer(fb.id);
            rlDisableFramebuffer();
            return FramebufferHandle();
        }

        int attachType = RL_ATTACHMENT_COLOR_CHANNEL0 + static_cast<int>(i);
        rlFramebufferAttach(fb.id, texId, attachType, RL_ATTACHMENT_CUBEMAP_POSITIVE_X, 0);

        FramebufferAttachment attachment;
        attachment.textureId = texId;
        attachment.format = colorFormats[i];
        attachment.attachType = attachType;
        fb.colorAttachments.push_back(attachment);
    }

    // Depth is a plain (non-cubemap) renderbuffer, faceSize x faceSize,
    // reused and re-cleared across all 6 faces -- see the comment on
    // Framebuffer::isCubemap for why this doesn't need to be a cubemap
    // itself.
    if (withDepth) {
        fb.depthRenderbufferId = rlLoadTextureDepth(faceSize, faceSize, true);
        if (fb.depthRenderbufferId == 0) {
            for (auto& att : fb.colorAttachments) rlUnloadTexture(att.textureId);
            rlUnloadFramebuffer(fb.id);
            rlDisableFramebuffer();
            return FramebufferHandle();
        }
        rlFramebufferAttach(fb.id, fb.depthRenderbufferId,
                            RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_RENDERBUFFER, 0);
    }

    if (!rlFramebufferComplete(fb.id)) {
        for (auto& att : fb.colorAttachments) rlUnloadTexture(att.textureId);
        if (fb.depthRenderbufferId) rlUnloadTexture(fb.depthRenderbufferId);
        rlUnloadFramebuffer(fb.id);
        return FramebufferHandle();
    }

    const std::string fbName = fb.name;
    framebuffers_.push_back(std::move(fb));
    framebuffersIndex_[fbName] = framebuffers_.size() - 1;
    return FramebufferHandle(fbName);
}

FramebufferHandle RenderGraph::CreateTextureArrayFramebuffer(const std::string& name,
                                              int width, int height, int layers,
                                              const std::vector<int>& colorFormats,
                                              bool withDepth) {
    if (name.empty() || FindFramebuffer(name) || width <= 0 || height <= 0 ||
        layers <= 0 || colorFormats.empty())
        return FramebufferHandle();

    Framebuffer fb;
    fb.name = name;
    fb.width = width;
    fb.height = height;
    fb.isTextureArray = true;
    fb.layerCount = layers;

    fb.id = rlLoadFramebuffer();
    if (fb.id == 0) return FramebufferHandle();

    rlEnableFramebuffer(fb.id);
    rlActiveDrawBuffers(static_cast<int>(colorFormats.size()));

    auto cleanupAndFail = [&]() -> FramebufferHandle {
        for (auto& att : fb.colorAttachments) rlUnloadTexture(att.textureId);
        if (fb.depthRenderbufferId) rlUnloadTexture(fb.depthRenderbufferId);
        rlUnloadFramebuffer(fb.id);
        rlDisableFramebuffer();
        return FramebufferHandle();
    };

    // No rlLoadTexture equivalent for GL_TEXTURE_2D_ARRAY, so this goes
    // through raw GL: glGenTextures + glTexImage3D allocate all `layers`
    // slices on one texture object, uninitialized. glFramebufferTextureLayer
    // is likewise the only way to point an FBO at one slice of it --
    // rlFramebufferAttach's texType enum has no array-layer option (only
    // TEXTURE2D/RENDERBUFFER/the 6 CUBEMAP faces).
    for (size_t i = 0; i < colorFormats.size(); ++i) {
        unsigned int internalFormat, format, type;
        if (!PixelFormatToGL(colorFormats[i], internalFormat, format, type))
            return cleanupAndFail();

        unsigned int texId = 0;
        glGenTextures(1, &texId);
        if (texId == 0) return cleanupAndFail();

        glBindTexture(GL_TEXTURE_2D_ARRAY, texId);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, static_cast<int>(internalFormat),
                     width, height, layers, 0, format, type, nullptr);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

        // Attach slice 0 for now, purely so rlFramebufferComplete below
        // has something valid to check -- GeometryPass execution
        // re-points this per-draw to whatever targetArrayLayer says
        // (see the Geometry branch in Apply()).
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + static_cast<int>(i), texId, 0, 0);

        FramebufferAttachment attachment;
        attachment.textureId = texId;
        attachment.format = colorFormats[i];
        attachment.attachType = RL_ATTACHMENT_COLOR_CHANNEL0 + static_cast<int>(i);
        fb.colorAttachments.push_back(attachment);
    }

    // Depth is a plain (non-array) renderbuffer at width x height,
    // reused and re-cleared across every layer -- same reasoning as the
    // cubemap path's depth attachment: it's only needed for depth
    // testing during the draw, never sampled back afterward.
    if (withDepth) {
        fb.depthRenderbufferId = rlLoadTextureDepth(width, height, true);
        if (fb.depthRenderbufferId == 0) return cleanupAndFail();
        rlFramebufferAttach(fb.id, fb.depthRenderbufferId,
                            RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_RENDERBUFFER, 0);
    }

    if (!rlFramebufferComplete(fb.id)) return cleanupAndFail();

    const std::string fbName = fb.name;
    framebuffers_.push_back(std::move(fb));
    framebuffersIndex_[fbName] = framebuffers_.size() - 1;
    return FramebufferHandle(fbName);
}

bool RenderGraph::DestroyFramebuffer(const std::string& name) {
    auto it = std::find_if(framebuffers_.begin(), framebuffers_.end(),
                           [&](const Framebuffer& fb) { return fb.name == name; });
    if (it == framebuffers_.end()) return false;

    // Block deletion if any geometry pass references this framebuffer
    for (const GeometryPass& pass : geometryPasses_)
        if (pass.targetFramebufferName == name) return false;

    // Unload all attached resources
    for (auto& att : it->colorAttachments) {
        if (att.textureId != 0) rlUnloadTexture(att.textureId);
    }
    if (it->depthRenderbufferId != 0) rlUnloadTexture(it->depthRenderbufferId);
    if (it->depthTextureId != 0) rlUnloadTexture(it->depthTextureId);
    if (it->id != 0) rlUnloadFramebuffer(it->id);

    framebuffers_.erase(it);
    RebuildNameIndex(framebuffers_, framebuffersIndex_);
    return true;
}
bool RenderGraph::ResizeFramebuffer(const std::string& name, int width, int height) {
    Framebuffer* fb = FindFramebuffer(name);
    if (!fb || width <= 0 || height <= 0) return false;
    // Neither path below is cubemap/array-aware (rlLoadTexture -- plain
    // 2D only). Rather than silently rebuild a cubemap or array target
    // as a broken 2D one, refuse outright until this function actually
    // grows that support.
    if (fb->isCubemap || fb->isTextureArray) return false;

    // Store original formats and depth presence
    std::vector<int> formats;
    formats.reserve(fb->colorAttachments.size());
    for (const auto& att : fb->colorAttachments) formats.push_back(att.format);
    bool withDepth = (fb->depthRenderbufferId != 0) || (fb->depthTextureId != 0);
    bool depthAsTexture = fb->depthTextureId != 0;

    // Create a temporary framebuffer with the new dimensions
    Framebuffer newFb;
    newFb.name = name;
    newFb.width = width;
    newFb.height = height;

    newFb.id = rlLoadFramebuffer();
    if (newFb.id == 0) return false;

    rlEnableFramebuffer(newFb.id);
    rlActiveDrawBuffers(static_cast<int>(formats.size()));

    // Attach color textures
    for (size_t i = 0; i < formats.size(); ++i) {
        unsigned int texId = rlLoadTexture(nullptr, width, height, formats[i], 1);
        if (texId == 0) {
            for (auto& att : newFb.colorAttachments) rlUnloadTexture(att.textureId);
            if (newFb.depthRenderbufferId) rlUnloadTexture(newFb.depthRenderbufferId);
            if (newFb.depthTextureId) rlUnloadTexture(newFb.depthTextureId);
            rlUnloadFramebuffer(newFb.id);
            rlDisableFramebuffer();
            return false;
        }
        int attachType = RL_ATTACHMENT_COLOR_CHANNEL0 + static_cast<int>(i);
        rlFramebufferAttach(newFb.id, texId, attachType, RL_ATTACHMENT_TEXTURE2D, 0);
        newFb.colorAttachments.push_back({texId, formats[i], attachType});
    }

    // Attach depth if original had one, preserving whether it was a
    // renderbuffer or a sampleable texture.
    if (withDepth) {
        if (depthAsTexture) {
            newFb.depthTextureId = rlLoadTextureDepth(width, height, false);
            if (newFb.depthTextureId == 0) {
                for (auto& att : newFb.colorAttachments) rlUnloadTexture(att.textureId);
                rlUnloadFramebuffer(newFb.id);
                rlDisableFramebuffer();
                return false;
            }
            rlFramebufferAttach(newFb.id, newFb.depthTextureId,
                                RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0);
        } else {
            newFb.depthRenderbufferId = rlLoadTextureDepth(width, height, true);
            if (newFb.depthRenderbufferId == 0) {
                for (auto& att : newFb.colorAttachments) rlUnloadTexture(att.textureId);
                rlUnloadFramebuffer(newFb.id);
                rlDisableFramebuffer();
                return false;
            }
            rlFramebufferAttach(newFb.id, newFb.depthRenderbufferId,
                                RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_RENDERBUFFER, 0);
        }
    }

    // Verify completeness
    if (!rlFramebufferComplete(newFb.id)) {
        for (auto& att : newFb.colorAttachments) rlUnloadTexture(att.textureId);
        if (newFb.depthRenderbufferId) rlUnloadTexture(newFb.depthRenderbufferId);
        if (newFb.depthTextureId) rlUnloadTexture(newFb.depthTextureId);
        rlUnloadFramebuffer(newFb.id);
        return false;
    }

    // Success – unload old resources and replace
    for (auto& att : fb->colorAttachments) if (att.textureId) rlUnloadTexture(att.textureId);
    if (fb->depthRenderbufferId) rlUnloadTexture(fb->depthRenderbufferId);
    if (fb->depthTextureId) rlUnloadTexture(fb->depthTextureId);
    if (fb->id) rlUnloadFramebuffer(fb->id);

    *fb = std::move(newFb);
    return true;
}
bool RenderGraph::BlitFramebuffer(const std::string& srcName, Rectangle srcRect,
                                  const std::string& dstName, Rectangle dstRect,
                                  int mask) {
    unsigned int srcId = 0;  // default framebuffer
    unsigned int dstId = 0;

    if (!srcName.empty()) {
        Framebuffer* src = FindFramebuffer(srcName);
        if (!src || src->id == 0) return false;
        srcId = src->id;
    }

    if (!dstName.empty()) {
        Framebuffer* dst = FindFramebuffer(dstName);
        if (!dst || dst->id == 0) return false;
        dstId = dst->id;
    }

    rlBindFramebuffer(RL_READ_FRAMEBUFFER, srcId);
    rlBindFramebuffer(RL_DRAW_FRAMEBUFFER, dstId);
    rlBlitFramebuffer(
        (int)srcRect.x, (int)srcRect.y,
        (int)(srcRect.x + srcRect.width), (int)(srcRect.y + srcRect.height),
        (int)dstRect.x, (int)dstRect.y,
        (int)(dstRect.x + dstRect.width), (int)(dstRect.y + dstRect.height),
        mask);
    rlDisableFramebuffer();
    return true;
}
Framebuffer* RenderGraph::FindFramebuffer(const std::string& name) {
    auto mapIt = framebuffersIndex_.find(name);
    if (mapIt != framebuffersIndex_.end() && mapIt->second < framebuffers_.size() &&
        framebuffers_[mapIt->second].name == name) {
        return &framebuffers_[mapIt->second];
    }
    auto it = std::find_if(framebuffers_.begin(), framebuffers_.end(),
                           [&](const Framebuffer& fb) { return fb.name == name; });
    return it == framebuffers_.end() ? nullptr : &(*it);
}

const Framebuffer* RenderGraph::FindFramebuffer(const std::string& name) const {
    auto mapIt = framebuffersIndex_.find(name);
    if (mapIt != framebuffersIndex_.end() && mapIt->second < framebuffers_.size() &&
        framebuffers_[mapIt->second].name == name) {
        return &framebuffers_[mapIt->second];
    }
    auto it = std::find_if(framebuffers_.begin(), framebuffers_.end(),
                           [&](const Framebuffer& fb) { return fb.name == name; });
    return it == framebuffers_.end() ? nullptr : &(*it);
}

void RenderGraph::BindFramebuffer(const std::string& name) {
    Framebuffer* fb = FindFramebuffer(name);
    if (fb && fb->id != 0) {
        rlEnableFramebuffer(fb->id);
    }
}

void RenderGraph::UnbindFramebuffer() {
    rlDisableFramebuffer();
}
bool RenderGraph::SetGeometryFloat(const std::string& passName, const std::string& uniformName, float value) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_FLOAT, currentShaderId_);
}

bool RenderGraph::SetGeometryInt(const std::string& passName, const std::string& uniformName, int value) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_INT, currentShaderId_);
}

bool RenderGraph::SetGeometryVector2(const std::string& passName, const std::string& uniformName, Vector2 value) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC2, currentShaderId_);
}

bool RenderGraph::SetGeometryVector3(const std::string& passName, const std::string& uniformName, Vector3 value) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC3, currentShaderId_);
}

bool RenderGraph::SetGeometryVector4(const std::string& passName, const std::string& uniformName, Vector4 value) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC4, currentShaderId_);
}

bool RenderGraph::SetGeometryColor(const std::string& passName, const std::string& uniformName, Color value) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass || !pass->shaderDefinition) return false;
    return ApplyCachedUniform(*pass, uniformName, value, RL_SHADER_UNIFORM_VEC4, currentShaderId_);
}

// Same caching shape as ApplyCachedUniform (skip the GPU push if the value
// hasn't actually changed) but written by hand instead of going through
// that template -- see MatrixUniform's doc comment for why. Matrix has no
// operator== (same gap UniformValueEquals works around for Vector2/3/4),
// so equality here is a flat memcmp over its 16 floats; that's exact
// rather than epsilon-based, same as every other UniformValueEquals case.
bool RenderGraph::SetGeometryMatrix(const std::string& passName, const std::string& uniformName, const Matrix& value) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass || !pass->shaderDefinition) return false;

    auto it = std::find_if(pass->matrixUniforms.begin(), pass->matrixUniforms.end(),
                            [&](const MatrixUniform& u) { return u.name == uniformName; });

    MatrixUniform* uniform = nullptr;
    if (it != pass->matrixUniforms.end()) {
        if (std::memcmp(&it->value, &value, sizeof(Matrix)) == 0) {
            it->dirty = false;
            return true; // already this value on the GPU -- nothing to push
        }
        it->value = value;
        it->dirty = true;
        uniform = &(*it);
    } else {
        unsigned int programId = PassProgramId(*pass);
        if (programId == 0) return false;
        const int location = rlGetLocationUniform(programId, uniformName.c_str());
        if (location < 0) return false;
        pass->matrixUniforms.push_back(MatrixUniform{uniformName, location, value, true});
        uniform = &pass->matrixUniforms.back();
    }

    unsigned int programId = PassProgramId(*pass);
    if (programId == 0) return false;

    rlEnableShader(programId);
    rlSetUniformMatrix(uniform->location, uniform->value);
    // Deliberately not calling rlDisableShader() here -- matches raylib's
    // own SetShaderValueMatrix, which leaves the shader bound rather than
    // unbinding it ("Avoid reset another shader already binded" in
    // rmodels.c). SetGeometryMatrix is meant to be callable from inside a
    // GeometryPass's drawCallback -- Apply() already has this pass's
    // shader enabled when drawCallback runs, and whatever runs right
    // after (more of drawCallback, or a drawItemsProvider's glDrawArrays
    // calls) needs that shader to still be bound. Unlike ApplyCachedUniform,
    // this one doesn't touch currentShaderId_ either, since the actually-
    // bound program hasn't changed.
    uniform->dirty = false;
    return true;
}

bool RenderGraph::SetGeometryAttributelessDraw(const std::string& passName, PrimitiveMode mode) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;
    pass->attributelessDraw = true;
    pass->primitiveMode = mode;
    return true;
}

bool RenderGraph::SetGeometryDrawItemsProvider(const std::string& passName, GeometryDrawItemsProvider provider) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;
    pass->drawItemsProvider = std::move(provider);
    return true;
}

GeometryPassHandle RenderGraph::CreateGeometryPass(const std::string& name,
                                         const std::string& shaderName) {
    if (name.empty() || FindGeometryPass(name) || FindPass(name) || FindGroup(name))
        return GeometryPassHandle();

    auto shaderIt = std::find_if(shaders_.begin(), shaders_.end(),
        [&](const ProcessShader& shader) { return shader.name == shaderName; });
    if (shaderIt == shaders_.end() || shaderIt->shader.id == 0)
        return GeometryPassHandle();

    GeometryPass pass;
    pass.name = name;
    pass.shaderName = shaderName;
    pass.shaderDefinition = &(*shaderIt);
    // Geometry passes rasterize real 3D content into a framebuffer, so depth
    // testing is on by default (unlike ProcessPass/ComputePass, which stay
    // off). This is what let the deferred-rendering sample call
    // rlEnableDepthTest() once itself instead of the graph managing it;
    // callers that truly want it off can still SetGeometryDepthTest(false).
    pass.renderState.depthTest = true;
    pass.renderState.depthWrite = true;

    geometryPasses_.push_back(std::move(pass));
    geometryPassesIndex_[name] = geometryPasses_.size() - 1;
    return GeometryPassHandle(name);
}

bool RenderGraph::DestroyGeometryPass(const std::string& name) {
    auto it = std::find_if(geometryPasses_.begin(), geometryPasses_.end(),
        [&](const GeometryPass& pass) { return pass.name == name; });
    if (it == geometryPasses_.end()) return false;

    // Remove from graph if present
    renderOrder_.erase(
        std::remove_if(renderOrder_.begin(), renderOrder_.end(),
            [&](const RenderNode& node) {
                return node.type == RenderNodeType::Geometry && node.name == name;
            }),
        renderOrder_.end()
    );
    geometryPasses_.erase(it);
    RebuildNameIndex(geometryPasses_, geometryPassesIndex_);
    return true;
}

GeometryPass* RenderGraph::FindGeometryPass(const std::string& name) {
    auto mapIt = geometryPassesIndex_.find(name);
    if (mapIt != geometryPassesIndex_.end() && mapIt->second < geometryPasses_.size() &&
        geometryPasses_[mapIt->second].name == name) {
        return &geometryPasses_[mapIt->second];
    }
    auto it = std::find_if(geometryPasses_.begin(), geometryPasses_.end(),
        [&](const GeometryPass& pass) { return pass.name == name; });
    return it == geometryPasses_.end() ? nullptr : &(*it);
}

const GeometryPass* RenderGraph::FindGeometryPass(const std::string& name) const {
    auto mapIt = geometryPassesIndex_.find(name);
    if (mapIt != geometryPassesIndex_.end() && mapIt->second < geometryPasses_.size() &&
        geometryPasses_[mapIt->second].name == name) {
        return &geometryPasses_[mapIt->second];
    }
    auto it = std::find_if(geometryPasses_.begin(), geometryPasses_.end(),
        [&](const GeometryPass& pass) { return pass.name == name; });
    return it == geometryPasses_.end() ? nullptr : &(*it);
}
bool RenderGraph::AttachGeometryPassToGraph(const std::string& passName) {
    if (!FindGeometryPass(passName)) return false;

    auto it = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::Geometry && node.name == passName;
        });
    if (it != renderOrder_.end()) return false;

    renderOrder_.push_back({RenderNodeType::Geometry, passName});
    return true;
}

bool RenderGraph::RemoveGeometryPassFromGraph(const std::string& passName) {
    auto it = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::Geometry && node.name == passName;
        });
    if (it == renderOrder_.end()) return false;

    renderOrder_.erase(it);
    return true;
}

bool RenderGraph::InsertGeometryPassBefore(const std::string& passName,
                                           const std::string& beforeNodeName) {
    if (!FindGeometryPass(passName)) return false;

    auto beforeIt = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) { return node.name == beforeNodeName; });
    if (beforeIt == renderOrder_.end()) return false;

    auto geoIt = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::Geometry && node.name == passName;
        });
    if (geoIt != renderOrder_.end()) return false;

    renderOrder_.insert(beforeIt, {RenderNodeType::Geometry, passName});
    return true;
}

bool RenderGraph::InsertGeometryPassAfter(const std::string& passName,
                                          const std::string& afterNodeName) {
    if (!FindGeometryPass(passName)) return false;

    auto afterIt = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) { return node.name == afterNodeName; });
    if (afterIt == renderOrder_.end()) return false;

    auto geoIt = std::find_if(renderOrder_.begin(), renderOrder_.end(),
        [&](const RenderNode& node) {
            return node.type == RenderNodeType::Geometry && node.name == passName;
        });
    if (geoIt != renderOrder_.end()) return false;

    renderOrder_.insert(afterIt + 1, {RenderNodeType::Geometry, passName});
    return true;
}

bool RenderGraph::SetGeometryPassTargetFramebuffer(const std::string& passName,
                                                   const std::string& framebufferName) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;
    if (!FindFramebuffer(framebufferName)) return false;  // validate existence

    pass->targetFramebufferName = framebufferName;
    return true;
}

bool RenderGraph::SetGeometryPassCallback(const std::string& passName,
                                          std::function<void()> callback) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;

    pass->drawCallback = std::move(callback);
    return true;
}

bool RenderGraph::SetGeometryPassCubemapCapture(const std::string& passName,
                                                 Camera3D* camera,
                                                 Vector3 probePosition,
                                                 float nearPlane,
                                                 float farPlane) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;

    // camera == nullptr is the documented way to turn capture mode back
    // off; isCubemapCapture also requires fb->isCubemap at Apply() time
    // (see the Geometry branch), so pointing a capture pass at a plain
    // 2D framebuffer by mistake just falls through to normal behavior
    // instead of corrupting anything.
    pass->isCubemapCapture = (camera != nullptr);
    pass->cubemapCamera = camera;
    pass->cubemapProbePosition = probePosition;
    pass->cubemapNearPlane = nearPlane;
    pass->cubemapFarPlane = farPlane;
    return true;
}

bool RenderGraph::SetGeometryPassTargetArrayLayer(const std::string& passName, int layer) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass || layer < 0) return false;

    pass->targetArrayLayer = layer;
    return true;
}

bool RenderGraph::SetGeometryPassEnabled(const std::string& passName, bool enabled) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;

    pass->enabled = enabled;
    return true;
}
bool RenderGraph::BindUniformBufferToGeometryPass(const std::string& passName,
                                                  const std::string& bufferName,
                                                  unsigned int bindingIndex) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;
    if (!FindUniformBuffer(bufferName)) return false;
    pass->uniformBufferBindings.erase(
        std::remove_if(pass->uniformBufferBindings.begin(), pass->uniformBufferBindings.end(),
                       [&](const UniformBufferBinding& b) { return b.bindingIndex == bindingIndex; }),
        pass->uniformBufferBindings.end());
    pass->uniformBufferBindings.push_back({bufferName, bindingIndex});
    return true;
}

bool RenderGraph::UnbindUniformBufferFromGeometryPass(const std::string& passName,
                                                      unsigned int bindingIndex) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;
    auto it = std::find_if(pass->uniformBufferBindings.begin(), pass->uniformBufferBindings.end(),
                           [&](const UniformBufferBinding& b) { return b.bindingIndex == bindingIndex; });
    if (it == pass->uniformBufferBindings.end()) return false;
    pass->uniformBufferBindings.erase(it);
    return true;
}
bool RenderGraph::ValidateGraph() const {
    bool valid = true;
    std::unordered_set<std::string> executedPasses;
    std::unordered_set<std::string> executedGroups;

    // Helper to check a source name resolves and is executed
    auto CheckSource = [&](const std::string& sourceName, const std::string& consumerName) {
        if (sourceName.empty() || sourceName == SceneSourceName()) return;
        if (executedGroups.find(sourceName) != executedGroups.end()) return;
        if (executedPasses.find(sourceName) != executedPasses.end()) return;
        // Not yet executed
        TraceLog(LOG_WARNING, "RenderGraph: '%s' uses source '%s' which is not executed before it",
                 consumerName.c_str(), sourceName.c_str());
        valid = false;
    };

    for (const RenderNode& node : renderOrder_) {
        if (node.type == RenderNodeType::Pass) {
            const ProcessPass* pass = FindPass(node.name);
            if (!pass) { valid = false; continue; }

            // Skip passes that belong to a group – they are validated within the group
            if (IsPassGrouped(pass->name)) continue;

            // Check texture bindings
            for (const auto& binding : pass->textureBindings) {
                CheckSource(binding.sourcePassName, pass->name);
            }
            executedPasses.insert(pass->name);
        }
        else if (node.type == RenderNodeType::Group) {
            const ProcessGroup* group = FindGroup(node.name);
            if (!group) { valid = false; continue; }

            // Check group source
            CheckSource(group->sourceName, group->name);

            // Walk passes inside the group in order
            std::unordered_set<std::string> groupLocalPasses;
            for (const std::string& passName : group->passNames) {
                const ProcessPass* pass = FindPass(passName);
                if (!pass) { valid = false; continue; }

                // Dependencies can come from group source, earlier group passes, or global executed sets
                for (const auto& binding : pass->textureBindings) {
                    const std::string& src = binding.sourcePassName;
                    if (src.empty() || src == SceneSourceName()) continue;
                    if (src == group->sourceName) continue;
                    if (groupLocalPasses.find(src) != groupLocalPasses.end()) continue;
                    if (executedPasses.find(src) != executedPasses.end() ||
                        executedGroups.find(src) != executedGroups.end()) continue;
                    TraceLog(LOG_WARNING, "RenderGraph: group pass '%s' uses source '%s' which is not available",
                             pass->name.c_str(), src.c_str());
                    valid = false;
                }
                groupLocalPasses.insert(pass->name);
            }

            executedGroups.insert(group->name);
        }
        else if (node.type == RenderNodeType::Compute) {
            const ComputePass* pass = FindComputePass(node.name);
            if (!pass) { valid = false; continue; }

            for (const auto& binding : pass->textureBindings) {
                CheckSource(binding.sourcePassName, pass->name);
            }
            for (const auto& binding : pass->imageBindings) {
                CheckSource(binding.sourcePassName, pass->name);
            }
        }
        else if (node.type == RenderNodeType::ComputeGroup) {
            const ComputeGroup* group = FindComputeGroup(node.name);
            if (!group) { valid = false; continue; }

            for (const std::string& passName : group->computePassNames) {
                const ComputePass* pass = FindComputePass(passName);
                if (!pass) { valid = false; continue; }

                for (const auto& binding : pass->textureBindings) {
                    CheckSource(binding.sourcePassName, pass->name);
                }
                for (const auto& binding : pass->imageBindings) {
                    CheckSource(binding.sourcePassName, pass->name);
                }
            }
        }
        else if (node.type == RenderNodeType::Geometry) {
            const GeometryPass* pass = FindGeometryPass(node.name);
            if (!pass) { valid = false; continue; }

            // Ensure the target framebuffer exists
            if (!FindFramebuffer(pass->targetFramebufferName)) {
                TraceLog(LOG_WARNING, "RenderGraph: geometry pass '%s' targets non-existent framebuffer '%s'",
                         pass->name.c_str(), pass->targetFramebufferName.c_str());
                valid = false;
            }
        }
    }

    return valid;
}

std::vector<ValidationIssue> RenderGraph::Validate() const {
    std::vector<ValidationIssue> issues;

    auto addIssue = [&](const char* nodeType, const std::string& nodeName,
                         const char* referenceKind, const std::string& referencedName,
                         std::string message) {
        issues.push_back({nodeType, nodeName, referenceKind, referencedName, std::move(message)});
    };

    // A texture/image binding's source is valid if it's the scene, the
    // history sentinel, empty (unbound), or resolves to any node kind
    // that can actually produce an output. Order doesn't matter here --
    // that's ValidateGraph()'s job -- only existence does.
    auto sourceExists = [&](const std::string& src) {
        if (src.empty() || src == SceneSourceName() || src == HistorySourceName()) return true;
        return FindPass(src) || FindGroup(src) || FindComputePass(src) ||
               FindGeometryPass(src) || FindComputeGroup(src);
    };
    auto nodeExists = [&](const std::string& n) {
        return FindPass(n) || FindGroup(n) || FindComputePass(n) ||
               FindGeometryPass(n) || FindComputeGroup(n);
    };

    // -- Post-process passes --
    for (const ProcessPass& pass : passes_) {
        if (!pass.shaderDefinition) {
            addIssue("Pass", pass.name, "shader", pass.shaderName,
                     "Pass '" + pass.name + "' references shader '" + pass.shaderName + "' which failed to resolve");
        }
        if (!pass.targetFramebufferName.empty() && !FindFramebuffer(pass.targetFramebufferName)) {
            addIssue("Pass", pass.name, "target framebuffer", pass.targetFramebufferName,
                     "Pass '" + pass.name + "' targets framebuffer '" + pass.targetFramebufferName + "' which does not exist");
        }
        for (const TextureBinding& binding : pass.textureBindings) {
            if (!sourceExists(binding.sourcePassName)) {
                addIssue("Pass", pass.name, "texture source", binding.sourcePassName,
                         "Pass '" + pass.name + "' texture binding '" + binding.name + "' sources from '" +
                         binding.sourcePassName + "' which does not exist");
            }
        }
        for (const UniformBufferBinding& binding : pass.uniformBufferBindings) {
            if (!FindUniformBuffer(binding.bufferName)) {
                addIssue("Pass", pass.name, "uniform buffer", binding.bufferName,
                         "Pass '" + pass.name + "' binds uniform buffer '" + binding.bufferName + "' which does not exist");
            }
        }
    }

    // -- Geometry passes --
    for (const GeometryPass& pass : geometryPasses_) {
        if (!pass.shaderDefinition) {
            addIssue("GeometryPass", pass.name, "shader", pass.shaderName,
                     "Geometry pass '" + pass.name + "' references shader '" + pass.shaderName + "' which failed to resolve");
        }
        if (pass.targetFramebufferName.empty() || !FindFramebuffer(pass.targetFramebufferName)) {
            addIssue("GeometryPass", pass.name, "target framebuffer", pass.targetFramebufferName,
                     "Geometry pass '" + pass.name + "' targets framebuffer '" + pass.targetFramebufferName + "' which does not exist");
        }
        for (const ComputeBufferBinding& binding : pass.bufferBindings) {
            if (!FindBuffer(binding.bufferName)) {
                addIssue("GeometryPass", pass.name, "buffer", binding.bufferName,
                         "Geometry pass '" + pass.name + "' binds buffer '" + binding.bufferName + "' which does not exist");
            }
        }
        for (const UniformBufferBinding& binding : pass.uniformBufferBindings) {
            if (!FindUniformBuffer(binding.bufferName)) {
                addIssue("GeometryPass", pass.name, "uniform buffer", binding.bufferName,
                         "Geometry pass '" + pass.name + "' binds uniform buffer '" + binding.bufferName + "' which does not exist");
            }
        }
    }

    // -- Compute passes --
    for (const ComputePass& pass : computePasses_) {
        if (!pass.programDefinition) {
            addIssue("ComputePass", pass.name, "program", pass.programName,
                     "Compute pass '" + pass.name + "' references program '" + pass.programName + "' which failed to resolve");
        }
        for (const TextureBinding& binding : pass.textureBindings) {
            if (!sourceExists(binding.sourcePassName)) {
                addIssue("ComputePass", pass.name, "texture source", binding.sourcePassName,
                         "Compute pass '" + pass.name + "' texture binding '" + binding.name + "' sources from '" +
                         binding.sourcePassName + "' which does not exist");
            }
            if (!binding.kernelRegistryName.empty() &&
                kernelSamplerRegistry_.find(binding.kernelRegistryName) == kernelSamplerRegistry_.end()) {
                addIssue("ComputePass", pass.name, "sampler registry", binding.kernelRegistryName,
                         "Compute pass '" + pass.name + "' texture binding '" + binding.name + "' expects '" +
                         binding.kernelRegistryName + "' in the sampler registry, but RegisterSampler/RegisterTexture was never called for it");
            }
        }
        for (const ComputeImageBinding& binding : pass.imageBindings) {
            if (!sourceExists(binding.sourcePassName)) {
                addIssue("ComputePass", pass.name, "image source", binding.sourcePassName,
                         "Compute pass '" + pass.name + "' image binding '" + binding.name + "' sources from '" +
                         binding.sourcePassName + "' which does not exist");
            }
            if (!binding.kernelRegistryName.empty() &&
                kernelImageRegistry_.find(binding.kernelRegistryName) == kernelImageRegistry_.end()) {
                addIssue("ComputePass", pass.name, "image registry", binding.kernelRegistryName,
                         "Compute pass '" + pass.name + "' image binding '" + binding.name + "' expects '" +
                         binding.kernelRegistryName + "' in the image registry, but RegisterImage/RegisterTexture was never called for it");
            }
        }
        for (const ComputeBufferBinding& binding : pass.bufferBindings) {
            // binding.bufferName may be a literal buffer or a kernel block
            // name resolved through kernelBufferRegistry_ -- see the SSBO
            // bind loop in DispatchComputePass. Either counts as resolved.
            bool resolved = FindBuffer(binding.bufferName) != nullptr;
            if (!resolved) {
                auto regIt = kernelBufferRegistry_.find(binding.bufferName);
                resolved = regIt != kernelBufferRegistry_.end() && FindBuffer(regIt->second.name) != nullptr;
            }
            if (!resolved) {
                addIssue("ComputePass", pass.name, "buffer", binding.bufferName,
                         "Compute pass '" + pass.name + "' binds buffer '" + binding.bufferName +
                         "' which does not exist as a buffer, and is not a registered kernel buffer name either");
            }
        }
        for (const UniformBufferBinding& binding : pass.uniformBufferBindings) {
            if (!FindUniformBuffer(binding.bufferName)) {
                addIssue("ComputePass", pass.name, "uniform buffer", binding.bufferName,
                         "Compute pass '" + pass.name + "' binds uniform buffer '" + binding.bufferName + "' which does not exist");
            }
        }
    }

    // -- Groups --
    for (const ProcessGroup& group : groups_) {
        if (!group.sourceName.empty() && !sourceExists(group.sourceName)) {
            addIssue("Group", group.name, "group source", group.sourceName,
                     "Group '" + group.name + "' sources from '" + group.sourceName + "' which does not exist");
        }
        for (const std::string& passName : group.passNames) {
            if (!FindPass(passName)) {
                addIssue("Group", group.name, "group member", passName,
                         "Group '" + group.name + "' lists pass '" + passName + "' which does not exist");
            }
        }
    }

    // -- Compute groups --
    for (const ComputeGroup& group : computeGroups_) {
        for (const std::string& passName : group.computePassNames) {
            if (!FindComputePass(passName)) {
                addIssue("ComputeGroup", group.name, "group member", passName,
                         "Compute group '" + group.name + "' lists compute pass '" + passName + "' which does not exist");
            }
        }
    }

    // -- Dependencies -- (AddDependency() validates both ends at insertion
    // time, but nothing currently purges dependencies_ when a Destroy*() on
    // either endpoint runs afterward, so this catches ones that have since
    // gone stale.)
    for (const auto& [src, dst] : dependencies_) {
        if (!nodeExists(src)) {
            addIssue("Dependency", src + " -> " + dst, "dependency", src,
                     "Dependency references '" + src + "' which no longer exists");
        }
        if (!nodeExists(dst)) {
            addIssue("Dependency", src + " -> " + dst, "dependency", dst,
                     "Dependency references '" + dst + "' which no longer exists");
        }
    }

    return issues;
}

bool RenderGraph::AddDependency(const std::string& sourceName, const std::string& targetName) {
    if (sourceName.empty() || targetName.empty() || sourceName == targetName) return false;

    // Check existence across all node types, including compute groups.
    bool sourceExists = FindPass(sourceName) || FindGroup(sourceName) ||
                        FindComputePass(sourceName) || FindGeometryPass(sourceName) ||
                        FindComputeGroup(sourceName);
    bool targetExists = FindPass(targetName) || FindGroup(targetName) ||
                        FindComputePass(targetName) || FindGeometryPass(targetName) ||
                        FindComputeGroup(targetName);
    if (!sourceExists || !targetExists) return false;

    // Avoid duplicates
    for (const auto& [src, dst] : dependencies_)
        if (src == sourceName && dst == targetName) return false;
    dependencies_.emplace_back(sourceName, targetName);
    return true;
}
bool RenderGraph::RemoveDependency(const std::string& sourceName, const std::string& targetName) {
    auto it = std::find_if(dependencies_.begin(), dependencies_.end(),
        [&](const auto& dep) { return dep.first == sourceName && dep.second == targetName; });
    if (it == dependencies_.end()) return false;
    dependencies_.erase(it);
    return true;
}
void RenderGraph::SetAutomaticOrdering(bool enable) { automaticOrdering_ = enable; }
bool RenderGraph::GetAutomaticOrdering() const { return automaticOrdering_; }

std::string RenderGraph::NodeTypeToString(RenderNodeType type) {
    switch (type) {
        case RenderNodeType::Pass:     return "Pass";
        case RenderNodeType::Group:    return "Group";
        case RenderNodeType::Compute:  return "Compute";
        case RenderNodeType::ComputeGroup: return "Compute Group";
        case RenderNodeType::Geometry: return "Geometry";
    }
    return "Unknown";
}

bool RenderGraph::ExportGraphToDot(const std::string& filePath) const {
    if (filePath.empty()) {
        SetError("ExportGraphToDot: empty file path");
        return false;
    }

    std::vector<RenderNode> executionOrder;
    if (automaticOrdering_) {
        ComputeExecutionOrder(executionOrder);
    } else {
        executionOrder = renderOrder_;
    }

    std::ostringstream dot;
    dot << "digraph RenderGraph {\n";
    dot << "  rankdir=LR;\n";

    // Output all top-level nodes in execution order
    for (const RenderNode& node : executionOrder) {
        dot << "  \"" << node.name << "\" [shape=box, label=\""
            << node.name << " (" << NodeTypeToString(node.type) << ")\"];\n";
    }

    // Solid edges: explicit + inferred dependencies (unchanged)
    for (const auto& [src, dst] : dependencies_) {
        dot << "  \"" << src << "\" -> \"" << dst << "\";\n";
    }

    // Dashed edges: texture/image binding sources that reference another
    // pass/group by name (unchanged for existing types)
    auto emitSourceEdges = [&](const std::string& consumer, const std::vector<TextureBinding>& bindings) {
        for (const TextureBinding& binding : bindings) {
            if (binding.sourcePassName.empty() || binding.sourcePassName == consumer) continue;
            if (binding.sourcePassName == SceneSourceName() || binding.sourcePassName == HistorySourceName())
                continue;
            dot << "  \"" << binding.sourcePassName << "\" -> \"" << consumer
                << "\" [style=dashed];\n";
        }
    };

    // Post-process passes
    for (const ProcessPass& pass : passes_)
        emitSourceEdges(pass.name, pass.textureBindings);

    // Compute passes
    for (const ComputePass& pass : computePasses_) {
        emitSourceEdges(pass.name, pass.textureBindings);
        for (const ComputeImageBinding& binding : pass.imageBindings) {
            if (binding.sourcePassName.empty() || binding.sourcePassName == pass.name) continue;
            if (binding.sourcePassName == SceneSourceName()) continue;
            dot << "  \"" << binding.sourcePassName << "\" -> \"" << pass.name
                << "\" [style=dashed];\n";
        }
    }

    // Post-process groups
    for (const ProcessGroup& group : groups_) {
        if (!group.sourceName.empty() && group.sourceName != SceneSourceName()) {
            dot << "  \"" << group.sourceName << "\" -> \"" << group.name
                << "\" [style=dashed];\n";
        }
    }

    // ----- NEW: Compute groups -----
    for (const ComputeGroup& group : computeGroups_) {
        // Draw edges from internal compute passes to the group (or group to passes)
        // Here we show dashed edges from group to each internal pass to indicate containment.
        for (const std::string& passName : group.computePassNames) {
            dot << "  \"" << group.name << "\" -> \"" << passName
                << "\" [style=dotted, color=gray];\n";
        }
        // Also, if any internal pass has dependencies to external nodes,
        // those are already emitted in the compute pass section above.
    }

    dot << "}\n";

    const std::string content = dot.str();
    if (!SaveFileText(filePath.c_str(), const_cast<char*>(content.c_str()))) {
        SetError("ExportGraphToDot: failed to write '" + filePath + "'");
        return false;
    }
    return true;
}
bool RenderGraph::ImportDependencies(const std::string& filePath) {
    if (filePath.empty()) {
        SetError("LoadDependenciesFromFile: empty file path");
        return false;
    }
    if (!FileExists(filePath.c_str())) {
        SetError("LoadDependenciesFromFile: file not found '" + filePath + "'");
        return false;
    }

    char* raw = LoadFileText(filePath.c_str());
    if (!raw) {
        SetError("LoadDependenciesFromFile: failed to read '" + filePath + "'");
        return false;
    }
    std::string text(raw);
    UnloadFileText(raw);

    std::istringstream stream(text);
    std::string line;
    int lineNumber = 0;
    bool allValid = true;

    while (std::getline(stream, line)) {
        ++lineNumber;

        // Trim whitespace.
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;  // blank line
        auto last = line.find_last_not_of(" \t\r\n");
        line = line.substr(first, last - first + 1);

        if (line.empty() || line[0] == '#') continue;

        std::string source, target;
        auto arrowPos = line.find("->");
        auto commaPos = line.find(',');

        if (arrowPos != std::string::npos) {
            source = line.substr(0, arrowPos);
            target = line.substr(arrowPos + 2);
        } else if (commaPos != std::string::npos) {
            source = line.substr(0, commaPos);
            target = line.substr(commaPos + 1);
        } else {
            SetError("LoadDependenciesFromFile: malformed line " + std::to_string(lineNumber) +
                      " ('" + line + "'), expected 'source -> target' or 'source,target'");
            allValid = false;
            continue;
        }

        auto trim = [](std::string s) {
            auto f = s.find_first_not_of(" \t\r\n");
            if (f == std::string::npos) return std::string();
            auto l = s.find_last_not_of(" \t\r\n");
            return s.substr(f, l - f + 1);
        };
        source = trim(source);
        target = trim(target);

        if (!AddDependency(source, target)) {
            SetError("LoadDependenciesFromFile: invalid edge at line " + std::to_string(lineNumber) +
                      " ('" + source + "' -> '" + target + "')");
            allValid = false;
        }
    }

    return allValid;
}

void RenderGraph::ComputeExecutionOrder(std::vector<RenderNode>& outOrder) const {
    outOrder.clear();
    if (renderOrder_.empty()) return;

    // Map node name -> index in renderOrder_
    std::unordered_map<std::string, size_t> indexMap;
    for (size_t i = 0; i < renderOrder_.size(); ++i)
        indexMap[renderOrder_[i].name] = i;

    // Start with manually added dependencies
    std::vector<std::pair<std::string, std::string>> allDeps = dependencies_;

    // Helper to add a dependency if it is valid (source and target are top-level, not self, not duplicate)
    auto addInferredDependency = [&](const std::string& source, const std::string& target) {
        if (source.empty() || target.empty() || source == target) return;
        if (source == SceneSourceName()) return; // scene is always available
        // Only add if both source and target are top-level nodes
        if (indexMap.find(source) == indexMap.end() || indexMap.find(target) == indexMap.end())
            return;
        // Avoid duplicates
        for (const auto& dep : allDeps) {
            if (dep.first == source && dep.second == target) return;
        }
        allDeps.emplace_back(source, target);
    };

    // Iterate over top-level nodes and infer dependencies from bindings
    for (const RenderNode& node : renderOrder_) {
        if (node.type == RenderNodeType::Pass) {
            const ProcessPass* pass = FindPass(node.name);
            if (!pass) continue;
            // Skip passes that are part of a group (their ordering is handled inside the group)
            if (IsPassGrouped(pass->name)) continue;

            for (const TextureBinding& binding : pass->textureBindings) {
                if (!binding.sourcePassName.empty() && binding.sourcePassName != pass->name) {
                    addInferredDependency(binding.sourcePassName, pass->name);
                }
            }
        }
        else if (node.type == RenderNodeType::Compute) {
            const ComputePass* pass = FindComputePass(node.name);
            if (!pass) continue;

            for (const TextureBinding& binding : pass->textureBindings) {
                if (!binding.sourcePassName.empty() && binding.sourcePassName != pass->name) {
                    addInferredDependency(binding.sourcePassName, pass->name);
                }
            }
            for (const ComputeImageBinding& binding : pass->imageBindings) {
                if (!binding.sourcePassName.empty() && binding.sourcePassName != pass->name) {
                    addInferredDependency(binding.sourcePassName, pass->name);
                }
            }
        }
        else if (node.type == RenderNodeType::ComputeGroup) {
            const ComputeGroup* group = FindComputeGroup(node.name);
            if (!group) continue;
            for (const std::string& passName : group->computePassNames) {
                const ComputePass* pass = FindComputePass(passName);
                if (!pass) continue;
                for (const TextureBinding& binding : pass->textureBindings) {
                    if (!binding.sourcePassName.empty() && binding.sourcePassName != pass->name) {
                        addInferredDependency(binding.sourcePassName, node.name);
                    }
                }
                for (const ComputeImageBinding& binding : pass->imageBindings) {
                    if (!binding.sourcePassName.empty() && binding.sourcePassName != pass->name) {
                        addInferredDependency(binding.sourcePassName, node.name);
                    }
                }
            }
        }
        else if (node.type == RenderNodeType::Group) {
            const ProcessGroup* group = FindGroup(node.name);
            if (!group) continue;
            if (!group->sourceName.empty()) {
                addInferredDependency(group->sourceName, group->name);
            }
        }
        // Geometry passes have no texture/image bindings, so nothing to infer
    }

    // Build adjacency and in-degree using all dependencies (manual + inferred)
    std::unordered_map<std::string, std::vector<std::string>> adjacency;
    std::unordered_map<std::string, int> inDegree;
    for (const auto& node : renderOrder_)
        inDegree[node.name] = 0;

    for (const auto& [src, dst] : allDeps) {
        // Only consider dependencies where both nodes are in the render order
        if (indexMap.find(src) == indexMap.end() || indexMap.find(dst) == indexMap.end())
            continue;
        adjacency[src].push_back(dst);
        inDegree[dst]++;
    }

    // Priority queue: lower user-set priority first, then smallest original
    // index as the stable tie-breaker for nodes with equal (default 0)
    // priority.
    auto cmp = [&](const std::string& a, const std::string& b) {
        int pa = GetNodePriority(a);
        int pb = GetNodePriority(b);
        if (pa != pb) return pa > pb;
        return indexMap[a] > indexMap[b];
    };
    std::priority_queue<std::string, std::vector<std::string>, decltype(cmp)> queue(cmp);

    for (const auto& node : renderOrder_)
        if (inDegree[node.name] == 0)
            queue.push(node.name);

    std::vector<RenderNode> sorted;
    while (!queue.empty()) {
        std::string name = queue.top();
        queue.pop();
        auto it = std::find_if(renderOrder_.begin(), renderOrder_.end(),
            [&](const RenderNode& n) { return n.name == name; });
        if (it != renderOrder_.end())
            sorted.push_back(*it);

        for (const std::string& neighbor : adjacency[name]) {
            if (--inDegree[neighbor] == 0)
                queue.push(neighbor);
        }
    }

        // Cycle detected: fallback to original order and log the nodes involved
    if (sorted.size() != renderOrder_.size()) {
        outOrder = renderOrder_; // fallback

        // Collect names of nodes that are still part of a cycle (in-degree > 0 after processing)
        std::string cycleNodes;
        for (const auto& node : renderOrder_) {
            if (inDegree[node.name] > 0) {
                if (!cycleNodes.empty()) cycleNodes += ", ";
                cycleNodes += node.name;
            }
        }

        if (!cycleNodes.empty()) {
            TraceLog(LOG_WARNING, "RenderGraph: cycle detected in dependencies (including inferred). Nodes involved: %s. Using original order.", cycleNodes.c_str());
        } else {
            TraceLog(LOG_WARNING, "RenderGraph: cycle detected but no nodes remain with positive in-degree. Falling back to original order.");
        }
        return;
    }

    outOrder = std::move(sorted);
}
// ============================================================
// Uniform Buffer Object (UBO) implementation
// (raw OpenGL calls are required because rlgl does not expose UBO functions)
// ============================================================

UniformBufferHandle RenderGraph::CreateUniformBuffer(const std::string& name,
                                          unsigned int sizeBytes,
                                          const void* initialData,
                                          int usageHint) {
    if (name.empty() || sizeBytes == 0 || FindUniformBuffer(name))
        return UniformBufferHandle();

    unsigned int id = 0;
    glGenBuffers(1, &id);
    if (id == 0) return UniformBufferHandle();

    glBindBuffer(GL_UNIFORM_BUFFER, id);
    glBufferData(GL_UNIFORM_BUFFER, sizeBytes, initialData, usageHint);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    UniformBuffer buffer;
    buffer.name = name;
    buffer.id = id;
    buffer.size = sizeBytes;
    buffer.usageHint = usageHint;
    uniformBuffers_.push_back(buffer);
    return UniformBufferHandle(name);
}

bool RenderGraph::DestroyUniformBuffer(const std::string& name) {
    auto it = std::find_if(uniformBuffers_.begin(), uniformBuffers_.end(),
        [&](const UniformBuffer& b) { return b.name == name; });
    if (it == uniformBuffers_.end()) return false;

    // Block deletion if still bound to any pass
    for (const auto& pass : passes_)
        for (const auto& binding : pass.uniformBufferBindings)
            if (binding.bufferName == name) return false;
    for (const auto& pass : computePasses_)
        for (const auto& binding : pass.uniformBufferBindings)
            if (binding.bufferName == name) return false;

    if (it->id != 0) glDeleteBuffers(1, &it->id);
    uniformBuffers_.erase(it);
    return true;
}

bool RenderGraph::UpdateUniformBuffer(const std::string& name,
                                      const void* data,
                                      unsigned int sizeBytes,
                                      unsigned int offset) {
    UniformBuffer* buffer = FindUniformBuffer(name);
    if (!buffer || buffer->id == 0) return false;
    if (offset + sizeBytes > buffer->size) return false;

    glBindBuffer(GL_UNIFORM_BUFFER, buffer->id);
    glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeBytes, data);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    return true;
}

UniformBuffer* RenderGraph::FindUniformBuffer(const std::string& name) {
    auto it = std::find_if(uniformBuffers_.begin(), uniformBuffers_.end(),
        [&](const UniformBuffer& b) { return b.name == name; });
    return it == uniformBuffers_.end() ? nullptr : &(*it);
}

const UniformBuffer* RenderGraph::FindUniformBuffer(const std::string& name) const {
    auto it = std::find_if(uniformBuffers_.begin(), uniformBuffers_.end(),
        [&](const UniformBuffer& b) { return b.name == name; });
    return it == uniformBuffers_.end() ? nullptr : &(*it);
}

// Binding functions for ProcessPass
bool RenderGraph::BindUniformBufferToPass(const std::string& passName,
                                          const std::string& bufferName,
                                          unsigned int bindingIndex) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;
    if (!FindUniformBuffer(bufferName)) return false;

    // Replace existing binding at the same index
    pass->uniformBufferBindings.erase(
        std::remove_if(pass->uniformBufferBindings.begin(),
                       pass->uniformBufferBindings.end(),
                       [&](const UniformBufferBinding& b) {
                           return b.bindingIndex == bindingIndex;
                       }),
        pass->uniformBufferBindings.end());

    pass->uniformBufferBindings.push_back({bufferName, bindingIndex});
    return true;
}

bool RenderGraph::UnbindUniformBufferFromPass(const std::string& passName,
                                              unsigned int bindingIndex) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;
    auto it = std::find_if(pass->uniformBufferBindings.begin(),
                           pass->uniformBufferBindings.end(),
                           [&](const UniformBufferBinding& b) {
                               return b.bindingIndex == bindingIndex;
                           });
    if (it == pass->uniformBufferBindings.end()) return false;
    pass->uniformBufferBindings.erase(it);
    return true;
}

// Binding functions for ComputePass
bool RenderGraph::BindUniformBufferToComputePass(const std::string& passName,
                                                 const std::string& bufferName,
                                                 unsigned int bindingIndex) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    if (!FindUniformBuffer(bufferName)) return false;

    // Replace existing binding at the same index
    pass->uniformBufferBindings.erase(
        std::remove_if(pass->uniformBufferBindings.begin(),
                       pass->uniformBufferBindings.end(),
                       [&](const UniformBufferBinding& b) {
                           return b.bindingIndex == bindingIndex;
                       }),
        pass->uniformBufferBindings.end());

    pass->uniformBufferBindings.push_back({bufferName, bindingIndex});
    return true;
}

bool RenderGraph::UnbindUniformBufferFromComputePass(const std::string& passName,
                                                     unsigned int bindingIndex) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    auto it = std::find_if(pass->uniformBufferBindings.begin(),
                           pass->uniformBufferBindings.end(),
                           [&](const UniformBufferBinding& b) {
                               return b.bindingIndex == bindingIndex;
                           });
    if (it == pass->uniformBufferBindings.end()) return false;
    pass->uniformBufferBindings.erase(it);
    return true;
}
RenderTexture2D RenderGraph::AcquirePooledTexture(int width, int height, TextureFilter filter, int format, bool withDepth) {
    for (auto it = renderTexturePool_.begin(); it != renderTexturePool_.end(); ++it) {
        if (it->texture.texture.width == width && it->texture.texture.height == height &&
            it->format == format && it->withDepth == withDepth) {
            RenderTexture2D tex = it->texture;
            renderTexturePool_.erase(it);
            SetTextureFilter(tex.texture, filter);
            return tex;
        }
    }

    RenderTexture2D newTex;
    if (!AllocateRenderTargetWithFormat(newTex, width, height, format, withDepth)) return {};
    SetTextureFilter(newTex.texture, filter);
    return newTex;
}

void RenderGraph::ReleaseToPool(RenderTexture2D texture, int format, bool withDepth) {
    if (texture.id == 0) return;
    for (const auto& pooled : renderTexturePool_) {
        if (pooled.texture.id == texture.id) return;
    }
    renderTexturePool_.push_back({texture, format, withDepth});
}
void RenderGraph::EmitMemoryBarrier(unsigned int barrierBits) {
    // Insert memory barrier (choose bits based on available API). This is
    // the one place that knows glMemoryBarrier isn't guaranteed to be
    // link-time available on ES3/Android/Emscripten -- every call site
    // (texture/image barriers, SSBO barriers) routes through here instead
    // of repeating the #if guard.
#if defined(GRAPHICS_API_OPENGL_43)
    glMemoryBarrier(barrierBits);
#elif defined(GRAPHICS_API_OPENGL_ES3) && (__ANDROID__ || __EMSCRIPTEN__)
    typedef void (GL_APIENTRYP PFNGLMEMORYBARRIERPROC)(GLbitfield barriers);
    static PFNGLMEMORYBARRIERPROC glMemoryBarrier =
        (PFNGLMEMORYBARRIERPROC)rlGetProcAddress("glMemoryBarrier");
    if (glMemoryBarrier) {
        glMemoryBarrier(barrierBits);
    }
#else
    (void)barrierBits;
#endif
}

void RenderGraph::EnsureBarrierForTexture(unsigned int textureId) {
    if (textureId == 0) return;

    // Only if this texture was written by a compute pass earlier this frame
    if (computeWrittenTextures_.find(textureId) == computeWrittenTextures_.end())
        return;

    // Already flushed? skip
    if (barrierFlushedTextures_.find(textureId) != barrierFlushedTextures_.end())
        return;

    EmitMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    barrierFlushedTextures_.insert(textureId);
}

Texture2D RenderGraph::ResolveFramebufferColorTexture(const std::string& framebufferName, std::size_t attachmentIndex) const {
    const Framebuffer* fb = FindFramebuffer(framebufferName);
    if (!fb || fb->id == 0 || attachmentIndex >= fb->colorAttachments.size()) return Texture2D{};

    const FramebufferAttachment& color = fb->colorAttachments[attachmentIndex];
    if (color.textureId == 0) return Texture2D{};

    Texture2D tex{};
    tex.id = color.textureId;
    tex.width = fb->width;
    tex.height = fb->height;
    tex.mipmaps = 1;
    tex.format = color.format;
    return tex;
}
bool RenderGraph::SetPassCondition(const std::string& passName, std::function<bool()> condition) {
    ProcessPass* pass = FindPass(passName);
    if (!pass) return false;
    pass->condition = std::move(condition);
    return true;
}

bool RenderGraph::SetGroupCondition(const std::string& groupName, std::function<bool()> condition) {
    ProcessGroup* group = FindGroup(groupName);
    if (!group) return false;
    group->condition = std::move(condition);
    return true;
}

bool RenderGraph::SetComputePassCondition(const std::string& passName, std::function<bool()> condition) {
    ComputePass* pass = FindComputePass(passName);
    if (!pass) return false;
    pass->condition = std::move(condition);
    return true;
}

bool RenderGraph::SetGeometryPassCondition(const std::string& passName, std::function<bool()> condition) {
    GeometryPass* pass = FindGeometryPass(passName);
    if (!pass) return false;
    pass->condition = std::move(condition);
    return true;
}
void RenderGraph::SetProfilingEnabled(bool enabled) {
    profilingEnabled_ = enabled;
    if (!enabled) {
        gpuTimes_.clear();   // optionally clear previous times
    }
}

bool RenderGraph::IsProfilingEnabled() const {
    return profilingEnabled_;
}

float RenderGraph::GetGPUTime(const std::string& name) const {
    auto it = gpuTimes_.find(name);
    if (it != gpuTimes_.end()) return it->second;
    return 0.0f;
}

const std::unordered_map<std::string, float>& RenderGraph::GetGPUTimeMap() const {
    return gpuTimes_;
}