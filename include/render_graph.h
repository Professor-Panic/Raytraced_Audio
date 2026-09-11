#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "glad.h"

// ------------------------------------------------------------------
// Core uniform and buffer types
// ------------------------------------------------------------------

using UniformValue = std::variant<
    std::monostate,
    float,
    int,
    Vector2,
    Vector3,
    Vector4,
    Color
>;

struct Uniform {
    UniformValue value;
    std::string name;
    int location = -1;

    // True from the moment a Set* call stores a genuinely different value
    // until that value is pushed to the GPU (same call, in practice --
    // see StoreUniform/StoreComputeUniform). Exposed mainly for
    // inspector/debug UIs that want to show "this uniform changed last
    // frame"; the dirty-skip behavior itself lives in StoreUniform.
    bool dirty = false;

    Uniform() = default;
    Uniform(UniformValue initialValue, std::string uniformName, int cachedLocation)
        : value(std::move(initialValue)), name(std::move(uniformName)), location(cachedLocation) {}
};

// raylib doesn't give Vector2/Vector3/Vector4/Color an operator==, and
// std::variant's own operator== would need one to compare same-alternative
// values -- so this is the "one of my own" equality UniformValue actually
// needs. Two monostate values compare equal (both "unset"); values holding
// different alternatives are never equal.
inline bool UniformValueEquals(const UniformValue& a, const UniformValue& b) {
    if (a.index() != b.index()) return false;
    return std::visit([&](auto&& lhs) -> bool {
        using T = std::decay_t<decltype(lhs)>;
        const T& rhs = std::get<T>(b);
        if constexpr (std::is_same_v<T, std::monostate>) {
            return true;
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
            return lhs == rhs;
        } else if constexpr (std::is_same_v<T, Vector2>) {
            return lhs.x == rhs.x && lhs.y == rhs.y;
        } else if constexpr (std::is_same_v<T, Vector3>) {
            return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
        } else if constexpr (std::is_same_v<T, Vector4>) {
            return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
        } else if constexpr (std::is_same_v<T, Color>) {
            return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b && lhs.a == rhs.a;
        }
    }, a);
}

struct UniformBuffer {
    std::string name;
    unsigned int id = 0;     // OpenGL buffer object (GL_UNIFORM_BUFFER)
    unsigned int size = 0;   // size in bytes
    int usageHint = RL_DYNAMIC_DRAW;
};

struct UniformBufferBinding {
    std::string bufferName;
    unsigned int bindingIndex = 0; // matches layout(std140, binding = N) in shader
};

struct ComputeBufferBinding {
    std::string bufferName;
    unsigned int bindingIndex = 0; // matches layout(std430, binding = N) in the .glsl
};

// ------------------------------------------------------------------
// Texture and render target pooling
// ------------------------------------------------------------------

struct TextureBinding {
    std::string name;
    std::string uniformName;
    Texture2D texture{};
    int location = -1;
    std::string texturePath;
    std::string sourcePassName;
    // Non-empty when this binding was created by CreateComputePassFromKernel:
    // the name to look up in RenderGraph's kernel sampler registry at
    // dispatch time, every dispatch, instead of using `texture` or
    // `sourcePassName` directly. This is what makes RegisterSampler()
    // order-independent relative to CreateComputePassFromKernel -- the
    // lookup happens live, so it doesn't matter whether registration
    // happened before or after the pass was created, and re-registering
    // a different texture under the same name later takes effect on the
    // next dispatch too. Cleared by any explicit SetComputeTexture/
    // SetComputeTextureSource call, since that's the user overriding it.
    std::string kernelRegistryName;
    // True when `texture` (or whatever sourcePassName/kernelRegistryName
    // resolves to) is a GL_TEXTURE_CUBE_MAP rather than GL_TEXTURE_2D --
    // set automatically when the binding's source resolves to a cubemap
    // Framebuffer (see SetTextureSource / SetTextureSourceFramebufferAttachment).
    // Changes which glBindTexture target the sampler-binding loop uses;
    // the shader-side uniform is still declared samplerCube by the user,
    // same as any other texture binding.
    bool isCubemap = false;
    // Which color attachment to pull from when sourcePassName names a
    // Framebuffer with more than one color output (e.g. a G-buffer's
    // "normal" channel at index 1). Ignored for every other kind of
    // source. Set via SetTextureSourceFramebufferAttachment.
    std::size_t framebufferAttachmentIndex = 0;
    // True when the source is a GL_TEXTURE_2D_ARRAY (see
    // CreateTextureArrayFramebuffer) -- binds with glBindTexture(
    // GL_TEXTURE_2D_ARRAY, ...) instead of the 2D/cubemap paths. The
    // shader side declares sampler2DArray and picks a slice itself via
    // the sample coordinate's third component; there's no separate
    // "which layer" field here the way there is on GeometryPass, because
    // sampling doesn't need one -- the whole array is bound at once.
    bool isTextureArray = false;
    // True when `texture` is a genuine GL_TEXTURE_3D volume texture --
    // binds with glBindTexture(GL_TEXTURE_3D, ...) instead of the 2D/
    // cubemap/array paths. Unlike isCubemap/isTextureArray, there's no
    // Framebuffer path that produces this (you can't render into a
    // volume texture the way you can a 2D array's layers), so it's only
    // ever set by SetTexture3D, which hands in a raw externally-created
    // GL texture id -- same shape as how SetTexture hands in a plain 2D
    // Texture2D. Mutually exclusive with isCubemap/isTextureArray; set
    // one, and the others are cleared.
    bool isTexture3D = false;
};

struct PooledRenderTexture {
    RenderTexture2D texture;
    int format;
    bool withDepth;
};

// ------------------------------------------------------------------
// Shader / program types
// ------------------------------------------------------------------

struct ProcessShader {
    std::string name;
    std::string fragmentShaderPath;
    std::string vertexShaderPath;
    Shader shader{};
    // Kept as a raylib Shader (not a raw rlgl program id) intentionally --
    // see AddShader() in the .cpp for why. shader.id is what every rlgl
    // call below actually operates on; shader.locs is still populated by
    // raylib's loader and rlgl's batch renderer depends on those locations
    // existing (SHADER_LOC_MATRIX_MVP, SHADER_LOC_VERTEX_POSITION, etc.).
};

struct ShaderBuffer {
    std::string name;
    unsigned int id = 0;     // rlgl SSBO id, from rlLoadShaderBuffer
    unsigned int size = 0;   // size in bytes
    int usageHint = RL_DYNAMIC_COPY;
};

struct ComputeProgram {
    std::string name;
    std::string sourcePath;
    unsigned int id = 0;     // rlgl compute program id, from rlLoadComputeShaderProgram
};

struct ComputeImageBinding {
    std::string name;
    std::string uniformName;
    Texture2D texture{};
    int location = -1;
    unsigned int imageUnit = 0;
    int format = RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;  // raylib format (for regular textures)
    unsigned int glInternalFormat = 0;                 // OpenGL internal format (for raw textures)
    bool isRawTexture = false;                         // true if texture.id is a raw GL texture (3D/cubemap)
    bool readOnly = true;
    std::string sourcePassName;
    // Same idea as TextureBinding::kernelRegistryName, resolved against
    // the kernel *image* registry instead of the sampler one. See there
    // for the order-independence rationale.
    std::string kernelRegistryName;
};

// ------------------------------------------------------------------
// Pass types
// ------------------------------------------------------------------

// Fields common to every pass type (Process/Geometry/Compute). Split out
// as a base rather than merging the three pass types outright -- each
// still has materially different execution semantics (fullscreen quad vs.
// user draw calls vs. compute dispatch), so this only removes field
// duplication, not the type distinction. Public inheritance means every
// existing call site (pass.name, pass->uniforms, etc.) keeps working
// unchanged -- member access syntax doesn't change for a derived type.
struct PassCommon {
    std::string name;
    std::vector<std::string> classNames;
    std::function<bool()> condition;
    bool enabled = true;

    // Tie-breaker used by ComputeExecutionOrder when two nodes have no
    // dependency relation to each other. Lower runs earlier.
    int priority = 0;

    std::vector<Uniform> uniforms;
    std::vector<UniformBufferBinding> uniformBufferBindings;
};

// Blend/depth/stencil state a raster pass executes with. ExecutePass and
// the GeometryPass execution path used to hardcode rlDisableColorBlend()
// and leave depth/stencil untouched (forcing user code to call
// rlEnableDepthTest() etc. itself); this makes all three a per-pass choice
// instead.
enum class PassBlendMode {
    Disabled,   // rlDisableColorBlend() -- hard overwrite, no blending (previous default)
    Custom      // BeginBlendMode(customBlendMode) / EndBlendMode()
};

struct PassRenderState {
    PassBlendMode blendMode = PassBlendMode::Disabled;
    int customBlendMode = BLEND_ALPHA;   // used only when blendMode == Custom

    bool depthTest = false;
    bool depthWrite = true;              // only meaningful when depthTest is true
};

// Shared by pass types that bind a ProcessShader and carry a clear
// color/viewport/scissor for a framebuffer-shaped target (ProcessPass,
// GeometryPass) -- not ComputePass, which binds a ComputeProgram instead
// and has no per-shader clear/viewport concept.
struct ShaderBoundPass : PassCommon {
    std::string shaderName;
    ProcessShader* shaderDefinition = nullptr;
    Color clearColor = BLANK;
    Rectangle viewport{0, 0, 0, 0};
    Rectangle scissor{0, 0, 0, 0};
    PassRenderState renderState;
};

struct ProcessPass : ShaderBoundPass {
    std::string targetFramebufferName;   // if set, render to this framebuffer instead of own output
    std::vector<TextureBinding> textureBindings;
    bool useClipSpaceQuad = false;   // when true, draw fullscreen quad in clip-space
    int outputFormat = RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    bool outputWithDepth = true;
    bool persistentOutput = false;
    RenderTexture2D output{};

    float resolutionScale = 1.0f;
    TextureFilter outputFilter = TEXTURE_FILTER_BILINEAR;

    bool enableFeedback = false;
    RenderTexture2D feedback{};

    // History buffer: previous frame's output, resolvable via
    // RenderGraph::HistorySourceName() from any pass in the graph.
    bool keepHistory = false;
    RenderTexture2D history{};

    // When true, mipmaps are (re)generated for this pass's output texture
    // after it is rendered. Only applies to the normal (non-framebuffer)
    // output path.
    bool generateMipmaps = false;
};

// mat4 uniform, cached the same way Uniform/UniformValue cache float/int/
// vecN/Color -- kept as its own small struct instead of folding into
// UniformValue because rlSetUniform (the call ApplyCachedUniform/
// ApplyUniformRLGL funnel through) has no RL_SHADER_UNIFORM_MAT4 case;
// pushing a matrix goes through rlSetUniformMatrix instead, which takes
// the whole Matrix by value rather than a void* + count. Scoped to
// GeometryPass for now since that's the only pass type that's needed it
// so far -- lift it onto PassCommon if ComputePass/ProcessPass ever do.
struct MatrixUniform {
    std::string name;
    int location = -1;
    Matrix value{};
    bool dirty = false;
};

// What a GeometryPass draws with no bound vertex buffer -- the shader
// pulls per-vertex data out of an SSBO via gl_VertexID instead. Points is
// what lidar/particle-style SSBO-driven draws use; Triangles/Lines cover
// the same trick for procedural geometry.
enum class PrimitiveMode { Points, Triangles, Lines };

// One glDrawArrays call's worth of work for an attributeless (or regular)
// GeometryPass: which buffer goes at which SSBO binding index, and how
// many vertices to draw starting where. A GeometryDrawItemsProvider
// returns a list of these, re-evaluated fresh every Apply() call --
// unlike GeometryPass::bufferBindings (which is a *persistent* binding
// set once at setup time), nothing here is a standing binding, so it
// never blocks DestroyBuffer the way BindBufferToComputePass's
// persistent bindings can.
struct GeometryDrawItem {
    std::unordered_map<unsigned int, std::string> bufferBindings; // bindingIndex -> buffer name
    unsigned int vertexCount = 0;
    unsigned int firstVertex = 0;
};
using GeometryDrawItemsProvider = std::function<std::vector<GeometryDrawItem>()>;

struct GeometryPass : ShaderBoundPass {
    std::vector<ComputeBufferBinding> bufferBindings;
    std::string targetFramebufferName;
    std::function<void()> drawCallback;               // user-defined drawing code

    // mat4 uniforms set via SetGeometryMatrix -- see MatrixUniform above
    // for why these live in a separate vector instead of `uniforms`.
    std::vector<MatrixUniform> matrixUniforms;

    // When drawItemsProvider is set, Apply() calls it right after
    // drawCallback (once per drawCallback invocation -- so once per
    // cubemap face too, if this is a cubemap-capture pass) and issues one
    // glDrawArrays per returned GeometryDrawItem, binding each item's
    // buffers to their SSBO slots first. attributelessDraw/primitiveMode
    // control the VAO/point-mode state around that loop; leave
    // attributelessDraw false if drawCallback already bound a real vertex
    // buffer and you just want per-item SSBO binding without touching VAO
    // state.
    bool attributelessDraw = false;
    PrimitiveMode primitiveMode = PrimitiveMode::Points;
    GeometryDrawItemsProvider drawItemsProvider;

    // Sampler bindings available to drawCallback's shader while this pass
    // is bound (e.g. a sampler3D dither texture, a lookup table). Same
    // TextureBinding type as ProcessPass/ComputePass use, so isCubemap/
    // isTextureArray/isTexture3D all work the same way here -- see
    // CreateGeometryTextureBinding/SetGeometryTexture/SetGeometryTexture3D.
    // There is currently no SetGeometryTextureSource equivalent (pointing
    // this at another pass's or framebuffer's output) -- only a
    // directly-supplied Texture2D/raw id is supported for geometry passes
    // right now.
    std::vector<TextureBinding> textureBindings;

    // When true (the default), this pass clears its target framebuffer's
    // color+depth before drawing. Set to false on every pass after the
    // first when chaining several GeometryPasses into the same
    // targetFramebufferName (e.g. opaque -> water -> foliage) so later
    // passes draw on top of -- and correctly depth-test against -- what
    // earlier passes already wrote, instead of wiping it.
    bool clearTarget = true;

    // ------------------------------------------------------------------
    // Cubemap capture
    // ------------------------------------------------------------------
    // When true, Apply() runs drawCallback 6 times in a row instead of
    // once -- targetFramebufferName must name a Framebuffer created with
    // CreateCubemapFramebuffer. Before each of the 6 draws the graph
    // re-points every color attachment at the next cube face and mutates
    // *cubemapCamera to look down that face's axis from
    // cubemapProbePosition. drawCallback itself is untouched by any of
    // this -- it's still just BeginMode3D(*cam)/.../EndMode3D(), the same
    // callback shape as a normal (non-cubemap) GeometryPass. The caller
    // never issues an rlgl call to make this work.
    bool isCubemapCapture = false;
    Camera3D* cubemapCamera = nullptr;
    Vector3 cubemapProbePosition{};
    float cubemapNearPlane = 0.05f;
    float cubemapFarPlane = 100.0f;

    // Which slice of a texture-array targetFramebufferName this pass
    // writes into (see Framebuffer::isTextureArray). Meaningless -- and
    // ignored -- for a non-array target. Unlike cubemap capture, there's
    // no forced loop: this pass still draws exactly once per Apply()
    // call, into whichever layer this currently names. Change it with
    // SetGeometryPassTargetArrayLayer between frames (or between Apply()
    // calls, if you're amortizing a multi-layer fill across several) to
    // reach the other slices.
    int targetArrayLayer = 0;
};

struct ComputePass : PassCommon {
    std::string programName;
    ComputeProgram* programDefinition = nullptr;
    std::vector<ComputeBufferBinding> bufferBindings;
    std::vector<ComputeImageBinding> imageBindings;
    std::vector<TextureBinding> textureBindings;
    unsigned int groupsX = 1;
    unsigned int groupsY = 1;
    unsigned int groupsZ = 1;

    // If true, DispatchComputePass avoids redundantly re-enabling the shader
    // when it's already the currently-bound program (see currentShaderId_).
    // Bindings (buffers/images/uniforms) are still refreshed every dispatch.
    bool persistentState = false;
};

// The one thing SetFloat/SetComputeFloat/SetGeometryFloat (etc.) actually
// differ on is where the GL program id comes from. Overloading on pass
// type lets StoreUniform<T>() below be written once and work for all
// three instead of three near-identical copies.
inline unsigned int PassProgramId(const ShaderBoundPass& pass) {
    return pass.shaderDefinition ? pass.shaderDefinition->shader.id : 0;
}
inline unsigned int PassProgramId(const ComputePass& pass) {
    return pass.programDefinition ? pass.programDefinition->id : 0;
}

// ------------------------------------------------------------------
// Group types
// ------------------------------------------------------------------

struct ProcessGroup {
    std::string name;
    std::vector<std::string> passNames;
    std::function<bool()> condition;
    std::string sourceName;
    bool enabled = true;
    RenderTexture2D output{};
};

struct ProcessPassTemplate {
    std::string name;
    std::string shaderName;
};

struct ProcessGroupTemplate {
    std::string name;
    std::vector<ProcessPassTemplate> passes;
};

struct ProcessGroupInstance {
    std::string name;
    std::string templateName;
    std::string groupName;
};

struct ComputeGroup {
    std::string name;
    std::vector<std::string> computePassNames;
    std::function<bool()> condition;
    bool enabled = true;
    int priority = 0;  // for automatic ordering tie-breaker
};

// ------------------------------------------------------------------
// Framebuffer types
// ------------------------------------------------------------------

struct FramebufferAttachment {
    unsigned int textureId;
    int format;              // pixel format
    int attachType;          // color0, color1, ..., depth, stencil
};

struct Framebuffer {
    std::string name;
    unsigned int id = 0;
    int width = 0, height = 0;
    std::vector<FramebufferAttachment> colorAttachments;

    // Exactly one of these is nonzero when the framebuffer has a depth
    // attachment (both zero means no depth). Which one gets used is chosen
    // by CreateFramebuffer's depthAsTexture parameter:
    //  - depthRenderbufferId: a GL_RENDERBUFFER (the default) -- usable for
    //    depth testing while rendering into this framebuffer, but not
    //    sampleable from a shader afterward.
    //  - depthTextureId: a GL_DEPTH_COMPONENT texture -- also usable for
    //    depth testing, and can additionally be read back as a texture the
    //    same way a color attachment is (build a Texture2D from the id, the
    //    same pattern used for colorAttachments[i].textureId elsewhere;
    //    Texture2D::format isn't consulted on the graph's texture-binding
    //    path, only .id is, so leave it as a best-effort placeholder).
    unsigned int depthRenderbufferId = 0;
    unsigned int depthTextureId = 0;

    // True when this Framebuffer's color attachments are GL_TEXTURE_CUBE_MAP
    // textures (created by CreateCubemapFramebuffer) rather than plain 2D
    // ones. Each colorAttachments[i].textureId is still a single texture
    // object -- rlLoadTextureCubemap allocates all 6 faces on one id --
    // but which *face* the FBO is currently pointed at is not stored here;
    // it gets re-pointed via rlFramebufferAttach right before each of the
    // 6 draws in a cubemap-capture GeometryPass (see isCubemapCapture).
    bool isCubemap = false;
    int faceSize = 0;   // width == height == faceSize when isCubemap

    // True when this Framebuffer's color attachments are GL_TEXTURE_2D_ARRAY
    // textures (created by CreateTextureArrayFramebuffer). Mutually
    // exclusive with isCubemap. width/height are the per-layer size, same
    // meaning as for a plain 2D Framebuffer.
    bool isTextureArray = false;
    int layerCount = 0;
};

// ------------------------------------------------------------------
// Lightweight typed handles
// ------------------------------------------------------------------
// A Handle<Tag> wraps the *name* of a graph entity, not a raw index.
// Indices into passes_/shaders_/buffers_/etc. shift on every erase (see
// RebuildNameIndex() at each Destroy* call site), so an index-based handle
// would silently go stale -- or worse, silently start referring to a
// *different* entity -- the moment anything earlier in that same vector
// is Destroyd. Names are the graph's actual stable identity; a handle is
// a compile-time-typed wrapper around one, so passing the wrong *kind*
// of thing (a ComputePassHandle where a FramebufferHandle is expected)
// is a compile error instead of a typo that only surfaces at runtime.
//
// Every AddX(...) creation function hands back the corresponding handle
// directly -- store it if you want it, or ignore the return value the
// same as before (the entity is still created/looked up by name either
// way). You can also convert a name you already have via the GetXHandle(name)
// accessors on RenderGraph. Either way, pass the handle to the
// handle-typed overloads below instead of retyping the name at every
// call site. An invalid (default-constructed, or "no such entity" /
// "creation failed") handle has an empty name -- check with IsValid() /
// operator bool before use, same as checking a Find*() pointer against
// nullptr (AddX(...) failures that used to return InvalidPassIndex() now
// return an invalid handle the same way).
template <typename Tag>
struct Handle {
    std::string name;

    Handle() = default;
    explicit Handle(std::string entityName) : name(std::move(entityName)) {}

    bool IsValid() const { return !name.empty(); }
    explicit operator bool() const { return IsValid(); }

    bool operator==(const Handle& other) const { return name == other.name; }
    bool operator!=(const Handle& other) const { return name != other.name; }
};

struct PassTag {};
struct GeometryPassTag {};
struct ComputePassTag {};
struct ShaderTag {};
struct ComputeShaderTag {};
struct BufferTag {};
struct UniformBufferTag {};
struct FramebufferTag {};
struct GroupTag {};
struct ComputeGroupTag {};
struct TextureTag {};
using PassHandle          = Handle<PassTag>;
using GeometryPassHandle  = Handle<GeometryPassTag>;
using ComputePassHandle   = Handle<ComputePassTag>;
using ShaderHandle        = Handle<ShaderTag>;
using ComputeShaderHandle = Handle<ComputeShaderTag>;
using BufferHandle        = Handle<BufferTag>;
using UniformBufferHandle = Handle<UniformBufferTag>;
using FramebufferHandle   = Handle<FramebufferTag>;
using GroupHandle         = Handle<GroupTag>;
using ComputeGroupHandle  = Handle<ComputeGroupTag>;
using TextureHandle       = Handle<TextureTag>;

// ------------------------------------------------------------------
// Scoped binding handles (pass + binding name)
// ------------------------------------------------------------------
// Bindings are not globally unique; they live inside a pass. A scoped
// handle bundles the owning pass and the binding name so you can refer
// to a binding without retyping both strings at every call site.

struct PassTextureBindingHandle {
    PassHandle pass;
    std::string bindingName;

    PassTextureBindingHandle() = default;
    PassTextureBindingHandle(PassHandle p, std::string name)
        : pass(std::move(p)), bindingName(std::move(name)) {}

    bool IsValid() const { return pass.IsValid() && !bindingName.empty(); }
    explicit operator bool() const { return IsValid(); }

    bool operator==(const PassTextureBindingHandle& other) const {
        return pass == other.pass && bindingName == other.bindingName;
    }
    bool operator!=(const PassTextureBindingHandle& other) const {
        return !(*this == other);
    }
};

struct ComputeImageBindingHandle {
    ComputePassHandle pass;
    std::string bindingName;

    ComputeImageBindingHandle() = default;
    ComputeImageBindingHandle(ComputePassHandle p, std::string name)
        : pass(std::move(p)), bindingName(std::move(name)) {}

    bool IsValid() const { return pass.IsValid() && !bindingName.empty(); }
    explicit operator bool() const { return IsValid(); }

    bool operator==(const ComputeImageBindingHandle& other) const {
        return pass == other.pass && bindingName == other.bindingName;
    }
    bool operator!=(const ComputeImageBindingHandle& other) const {
        return !(*this == other);
    }
};

struct ComputeTextureBindingHandle {
    ComputePassHandle pass;
    std::string bindingName;

    ComputeTextureBindingHandle() = default;
    ComputeTextureBindingHandle(ComputePassHandle p, std::string name)
        : pass(std::move(p)), bindingName(std::move(name)) {}

    bool IsValid() const { return pass.IsValid() && !bindingName.empty(); }
    explicit operator bool() const { return IsValid(); }

    bool operator==(const ComputeTextureBindingHandle& other) const {
        return pass == other.pass && bindingName == other.bindingName;
    }
    bool operator!=(const ComputeTextureBindingHandle& other) const {
        return !(*this == other);
    }
};

struct GeometryTextureBindingHandle {
    GeometryPassHandle pass;
    std::string bindingName;

    GeometryTextureBindingHandle() = default;
    GeometryTextureBindingHandle(GeometryPassHandle p, std::string name)
        : pass(std::move(p)), bindingName(std::move(name)) {}

    bool IsValid() const { return pass.IsValid() && !bindingName.empty(); }
    explicit operator bool() const { return IsValid(); }

    bool operator==(const GeometryTextureBindingHandle& other) const {
        return pass == other.pass && bindingName == other.bindingName;
    }
    bool operator!=(const GeometryTextureBindingHandle& other) const {
        return !(*this == other);
    }
};

// ------------------------------------------------------------------
// Validation report
// ------------------------------------------------------------------
// One entry per broken reference found by RenderGraph::Validate(). Unlike
// ValidateGraph() (which walks the render order and checks execution-time
// dependencies), Validate() is order-independent: it checks every named
// reference on every entity -- texture/image sources, SSBO and uniform
// buffer bindings, target framebuffers, shader/program resolution, group
// membership, and AddDependency() entries -- against what actually exists,
// and returns the full list in one pass instead of surfacing problems one
// at a time as passes silently no-op at runtime.
struct ValidationIssue {
    std::string nodeType;       // "Pass", "GeometryPass", "ComputePass", "Group", "ComputeGroup", "Dependency"
    std::string nodeName;       // the entity that holds the broken reference
    std::string referenceKind;  // "texture source", "image source", "buffer", "uniform buffer",
                                 // "target framebuffer", "shader", "program", "group member", "dependency"
    std::string referencedName; // the name that failed to resolve
    std::string message;        // human-readable summary, ready to log as-is
};

// ------------------------------------------------------------------
// Profiling helper
// ------------------------------------------------------------------

namespace {
// RAII wrapper for GL_TIME_ELAPSED query
class QueryScope {
public:
    QueryScope(bool enabled, GLuint queryId) : enabled_(enabled), queryId_(queryId) {
        if (enabled_) glBeginQuery(GL_TIME_ELAPSED, queryId_);
    }
    ~QueryScope() {
        if (enabled_) glEndQuery(GL_TIME_ELAPSED);
    }
private:
    bool enabled_;
    GLuint queryId_;
};
}

// ------------------------------------------------------------------
// Shader kernel library (HLSL-style multi-kernel GLSL, auto-detected)
// ------------------------------------------------------------------
//
// Single-file, multi-kernel GLSL compute shaders, HLSL-style:
//
//   #version 430
//   struct Agent { vec2 position; float angle; };
//
//   layout(rg8, binding = 0) uniform image2D trailMap;
//   layout(binding = 2) uniform sampler2D prevtrailMap;
//   layout(std430, binding = 1) buffer AgentBuffer { Agent agents[]; };
//
//   // binding is optional on all three, HLSL-`register()`-style -- omit
//   // it and one gets auto-assigned (see BINDING ASSIGNMENT below):
//   layout(rg8) uniform image2D scratchMap;
//   uniform sampler2D noiseTex;
//   layout(std430) buffer ScoreBuffer { float scores[]; };
//
//   uniform int numAgents;
//   uniform float moveSpeed;
//   uniform float evaporateSpeed;
//
//   uint hash(uint state) { ... }              // ordinary helper, shared
//   float sense(vec2 pos, vec2 dir) { ... }    // ordinary helper, shared
//
//   [numthreads(64, 1, 1)]
//   void Agent_Pass() {
//       // uses trailMap, prevtrailMap, AgentBuffer, numAgents, moveSpeed
//   }
//
//   [numthreads(16, 16, 1)]
//   void Evaporate() {
//       // uses trailMap, evaporateSpeed only
//   }
//
// Everything that is NOT a `[numthreads(...)] void Name() { ... }` block
// is treated as shared code (kept verbatim, in order, for every kernel).
// For a given kernel, its own block is unwrapped -- the function body is
// dropped into a generated `void main() { ... }` -- and every other
// kernel's block is simply omitted from that compile unit.
//
// Resource reflection (images/textures/buffers/scalar uniforms) is
// parsed once from the shared code, then each kernel's binding lists
// are filtered down to only the resources whose name actually appears
// (as a whole word) inside that kernel's body -- so a pass only ends up
// asking to bind what it uses.
//
// AUTO-DETECTION: LoadFile/LoadSource look for at least one
// `[numthreads(...)] void Name() { ... }` block. If none is found, the
// source is treated as a single, ordinary GLSL compute shader (its own
// ordinary `void main()` and `layout(local_size_x=...) in;` left
// untouched) and reflection runs over the whole file instead of a
// kernel body -- there's nothing to filter against, since the one
// implicit "kernel" is the only thing in the file, so every resource it
// declares belongs to it. Both cases end up as ordinary KernelInfo
// entries, so everything downstream (RenderGraph::CreateComputePassFromKernel,
// resource binding, automatic uniforms) treats a hand-written GLSL
// compute shader and an HLSL-style multi-kernel one exactly the same way.
//
// BINDING ASSIGNMENT: `binding = N` is optional on image2D, sampler2D,
// and buffer declarations, the same way HLSL lets you skip `register()`.
// Parsing first collects every *explicit* binding for a given resource
// kind (images, samplers, and buffers each occupy their own namespace,
// matching separate GL binding spaces), then walks the declarations in
// source order and hands each one still unassigned the lowest index not
// already claimed -- explicit and auto-assigned bindings never collide,
// and two declarations of the same kind never end up sharing a slot. By
// the time a KernelInfo is handed back, every resource's `binding` field
// is a real, resolved index; -1 never escapes LoadFile/LoadSource. A
// sampler2D's binding is currently reflected for completeness but not
// enforced at dispatch time -- see TextureBindingInfo.
//
// This is regex + brace-counting, not a real GLSL parser. It assumes
// one declaration per line in the style shown above, and that kernel
// bodies don't contain string/char literals with unbalanced braces.

enum class UniformType { Int, Float, Vec2, Vec3, Vec4, Bool };

struct ScalarUniformInfo {
    std::string name;
    UniformType type;
};

struct ImageBindingInfo {
    std::string name;      // GLSL variable name, e.g. "trailMap"
    std::string format;    // e.g. "rg8", "rgba8"
    // Parsed from `layout(..., binding = N)` if present; otherwise
    // auto-assigned (lowest free image-unit index) once parsing of the
    // shared code finishes. Always >= 0 on a KernelInfo returned by
    // LoadFile/LoadSource -- -1 only ever appears transiently mid-parse.
    int binding = -1;
};

struct TextureBindingInfo {
    std::string name;      // e.g. "prevtrailMap"
    // Parsed from `layout(binding = N)` if present; otherwise
    // auto-assigned (lowest free sampler-slot index), same rule as
    // ImageBindingInfo::binding. Reflected for completeness but not
    // currently enforced at dispatch -- compute sampler units are
    // assigned by iteration order in DispatchComputePass instead.
    int binding = -1;
};

struct BufferBindingInfo {
    std::string blockName; // e.g. "AgentBuffer"
    // Parsed from `layout(std430, binding = N)` if present; otherwise
    // auto-assigned (lowest free SSBO binding index), same rule as
    // ImageBindingInfo::binding.
    int binding = -1;
};

struct KernelInfo {
    std::string name;
    std::string fullSource;      // shared code + this kernel's body as main(), ready to compile
    int localSizeX = 1, localSizeY = 1, localSizeZ = 1;

    // True when this KernelInfo came from an ordinary single-entry-point
    // GLSL file (no [numthreads] blocks found) rather than an HLSL-style
    // multi-kernel one. Informational only -- binding and uniform
    // handling doesn't need to branch on it, since reflection already
    // scoped everything correctly at parse time.
    bool isPlainGlsl = false;

    std::vector<ScalarUniformInfo> uniforms;
    std::vector<ImageBindingInfo> images;
    std::vector<TextureBindingInfo> textures;
    std::vector<BufferBindingInfo> buffers;

    // Dispatch-size helper: local_size comes from [numthreads(...)] (or,
    // for a plain GLSL file, its own layout(local_size_x=...) in;) so
    // callers can't get the divisor wrong.
    void ComputeGroupCounts(unsigned int itemsX, unsigned int itemsY, unsigned int itemsZ,
                             unsigned int& outX, unsigned int& outY, unsigned int& outZ) const {
        outX = (itemsX + localSizeX - 1) / localSizeX;
        outY = (itemsY + localSizeY - 1) / localSizeY;
        outZ = (itemsZ + localSizeZ - 1) / localSizeZ;
    }

    bool HasUniform(const std::string& uniformName) const {
        for (auto& u : uniforms) if (u.name == uniformName) return true;
        return false;
    }
};

class ShaderKernelLibrary {
public:
    // plainKernelName only matters when the file turns out to hold no
    // [numthreads] blocks (i.e. it's an ordinary GLSL compute shader);
    // it becomes that one KernelInfo's name. Defaults to the file's stem
    // ("blur.comp" -> "blur") when left empty.
    void LoadFile(const std::string& path, const std::string& plainKernelName = "") {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("ShaderKernelLibrary: cannot open " + path);
        std::stringstream ss;
        ss << file.rdbuf();
        std::string name = plainKernelName;
        if (name.empty()) {
            size_t slash = path.find_last_of("/\\");
            size_t dot = path.find_last_of('.');
            size_t start = (slash == std::string::npos) ? 0 : slash + 1;
            size_t end = (dot == std::string::npos || dot < start) ? path.size() : dot;
            name = path.substr(start, end - start);
        }
        LoadSource(ss.str(), name);
    }

    // plainKernelName only matters for an ordinary (non-multi-kernel)
    // GLSL source; it names the resulting KernelInfo. Defaults to "main".
    void LoadSource(const std::string& source, const std::string& plainKernelName = "") {
        struct RawKernelBlock {
            std::string name;
            int localX, localY, localZ;
            size_t blockStart, blockEnd; // [start, end) over `source`, covers "[numthreads...] void Name() { ... }"
            std::string body;            // text strictly inside the function's braces
        };

        static const std::regex kernelHeader(
            R"(\[\s*numthreads\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*\]\s*\r?\n\s*void\s+(\w+)\s*\(\s*\)\s*\{)");

        std::vector<RawKernelBlock> blocks;
        for (auto it = std::sregex_iterator(source.begin(), source.end(), kernelHeader);
             it != std::sregex_iterator(); ++it) {
            const std::smatch& m = *it;
            size_t openBrace = static_cast<size_t>(m.position(0)) + static_cast<size_t>(m.length(0)) - 1;
            size_t closeBrace = FindMatchingBrace(source, openBrace);
            if (closeBrace == std::string::npos)
                throw std::runtime_error("ShaderKernelLibrary: unbalanced braces in kernel " + m[4].str());

            RawKernelBlock block;
            block.name = m[4].str();
            block.localX = std::stoi(m[1].str());
            block.localY = std::stoi(m[2].str());
            block.localZ = std::stoi(m[3].str());
            block.blockStart = static_cast<size_t>(m.position(0));
            block.blockEnd = closeBrace + 1;
            block.body = source.substr(openBrace + 1, closeBrace - openBrace - 1);
            blocks.push_back(std::move(block));
        }

        kernels_.clear();

        if (blocks.empty()) {
            // Auto-detected: ordinary GLSL compute shader, not the
            // HLSL-style multi-kernel format. Reflection runs over the
            // whole file and everything declared is kept -- there's no
            // per-kernel body to filter against.
            KernelInfo info;
            info.name = !plainKernelName.empty() ? plainKernelName : "main";
            info.isPlainGlsl = true;
            ParseGlobalDeclarations(source, info.images, info.textures, info.buffers, info.uniforms);

            static const std::regex localSizeDecl(
                R"(layout\s*\(\s*local_size_x\s*=\s*(\d+)\s*,\s*local_size_y\s*=\s*(\d+)\s*,\s*local_size_z\s*=\s*(\d+)\s*\)\s*in\s*;)");
            std::smatch sizeMatch;
            if (std::regex_search(source, sizeMatch, localSizeDecl)) {
                info.localSizeX = std::stoi(sizeMatch[1].str());
                info.localSizeY = std::stoi(sizeMatch[2].str());
                info.localSizeZ = std::stoi(sizeMatch[3].str());
            }

            info.fullSource = source; // already a complete, compilable shader
            kernels_[info.name] = std::move(info);
            return;
        }

        // Shared code = everything NOT inside a kernel block, concatenated in order.
        std::string sharedCode;
        size_t cursor = 0;
        for (auto& b : blocks) {
            sharedCode += source.substr(cursor, b.blockStart - cursor);
            cursor = b.blockEnd;
        }
        sharedCode += source.substr(cursor);

        ParseGlobalDeclarations(sharedCode, globalImages_, globalTextures_, globalBuffers_, globalUniforms_);

        for (auto& b : blocks) {
            KernelInfo info;
            info.name = b.name;
            info.localSizeX = b.localX;
            info.localSizeY = b.localY;
            info.localSizeZ = b.localZ;

            std::ostringstream src;
            src << sharedCode
                << "\nlayout(local_size_x = " << b.localX
                << ", local_size_y = " << b.localY
                << ", local_size_z = " << b.localZ << ") in;\n"
                << "void main()\n{" << b.body << "}\n";
            info.fullSource = src.str();

            // Only keep the resources this kernel's body actually references.
            for (auto& img : globalImages_)
                if (UsesIdentifier(b.body, img.name)) info.images.push_back(img);
            for (auto& tex : globalTextures_)
                if (UsesIdentifier(b.body, tex.name)) info.textures.push_back(tex);
            for (auto& buf : globalBuffers_)
                if (UsesIdentifier(b.body, buf.blockName)) info.buffers.push_back(buf);
            for (auto& u : globalUniforms_)
                if (UsesIdentifier(b.body, u.name)) info.uniforms.push_back(u);

            kernels_[b.name] = std::move(info);
        }
    }

    const KernelInfo& GetKernel(const std::string& name) const {
        auto it = kernels_.find(name);
        if (it == kernels_.end())
            throw std::runtime_error("ShaderKernelLibrary: no kernel named " + name);
        return it->second;
    }

    bool HasKernel(const std::string& name) const { return kernels_.find(name) != kernels_.end(); }

    std::vector<std::string> KernelNames() const {
        std::vector<std::string> names;
        names.reserve(kernels_.size());
        for (auto& [name, info] : kernels_) names.push_back(name);
        return names;
    }

private:
    std::unordered_map<std::string, KernelInfo> kernels_;
    std::vector<ImageBindingInfo> globalImages_;
    std::vector<TextureBindingInfo> globalTextures_;
    std::vector<BufferBindingInfo> globalBuffers_;
    std::vector<ScalarUniformInfo> globalUniforms_;

    static size_t FindMatchingBrace(const std::string& text, size_t openBracePos) {
        int depth = 0;
        for (size_t i = openBracePos; i < text.size(); ++i) {
            if (text[i] == '{') depth++;
            else if (text[i] == '}') {
                depth--;
                if (depth == 0) return i;
            }
        }
        return std::string::npos;
    }

    static bool UsesIdentifier(const std::string& body, const std::string& identifier) {
        std::regex wordBoundary("\\b" + identifier + "\\b");
        return std::regex_search(body, wordBoundary);
    }

    // Fills in `binding` (lowest free index, in declaration order) for
    // every entry that came out of parsing without an explicit `binding =
    // N`. Explicit bindings are reserved first regardless of where in the
    // source they appear, so an auto-assigned resource can never collide
    // with one the shader author pinned by hand -- same guarantee HLSL
    // gives you when you mix explicit `register()` slots with implicit
    // ones. Template over any of the three BindingInfo structs; they all
    // expose a plain `int binding`.
    template <typename BindingInfoT>
    static void AssignAutoBindings(std::vector<BindingInfoT>& resources) {
        std::unordered_set<int> used;
        for (const auto& r : resources)
            if (r.binding >= 0) used.insert(r.binding);

        int next = 0;
        for (auto& r : resources) {
            if (r.binding >= 0) continue;
            while (used.count(next)) next++;
            r.binding = next;
            used.insert(next);
            next++;
        }
    }

    static void ParseGlobalDeclarations(const std::string& sharedCode,
                                         std::vector<ImageBindingInfo>& images,
                                         std::vector<TextureBindingInfo>& textures,
                                         std::vector<BufferBindingInfo>& buffers,
                                         std::vector<ScalarUniformInfo>& uniforms) {
        // The `binding = N` clause is optional in all three -- HLSL-style:
        // give one explicitly, or leave it out and let AssignAutoBindings
        // fill it in below. The format qualifier on image2D and the
        // std430 qualifier on buffer stay mandatory; those aren't binding
        // syntax, they're GLSL's memory-layout requirements.
        static const std::regex imageDecl(
            R"(layout\s*\(\s*(\w+)\s*(?:,\s*binding\s*=\s*(\d+)\s*)?\)\s*uniform\s+image2D\s+(\w+)\s*;)");
        static const std::regex textureDecl(
            R"((?:layout\s*\(\s*binding\s*=\s*(\d+)\s*\)\s*)?uniform\s+sampler2D\s+(\w+)\s*;)");
        static const std::regex bufferDecl(
            R"(layout\s*\(\s*std430\s*(?:,\s*binding\s*=\s*(\d+)\s*)?\)\s*buffer\s+(\w+))");
        static const std::regex scalarDecl(
            R"(^\s*uniform\s+(int|float|bool|vec2|vec3|vec4)\s+(\w+)\s*;)");

        for (auto it = std::sregex_iterator(sharedCode.begin(), sharedCode.end(), imageDecl);
             it != std::sregex_iterator(); ++it) {
            ImageBindingInfo img;
            img.format = (*it)[1].str();
            img.binding = (*it)[2].matched ? std::stoi((*it)[2].str()) : -1;
            img.name = (*it)[3].str();
            images.push_back(img);
        }
        for (auto it = std::sregex_iterator(sharedCode.begin(), sharedCode.end(), textureDecl);
             it != std::sregex_iterator(); ++it) {
            TextureBindingInfo tex;
            tex.binding = (*it)[1].matched ? std::stoi((*it)[1].str()) : -1;
            tex.name = (*it)[2].str();
            textures.push_back(tex);
        }
        for (auto it = std::sregex_iterator(sharedCode.begin(), sharedCode.end(), bufferDecl);
             it != std::sregex_iterator(); ++it) {
            BufferBindingInfo buf;
            buf.binding = (*it)[1].matched ? std::stoi((*it)[1].str()) : -1;
            buf.blockName = (*it)[2].str();
            buffers.push_back(buf);
        }

        AssignAutoBindings(images);
        AssignAutoBindings(textures);
        AssignAutoBindings(buffers);

        std::stringstream ss(sharedCode);
        std::string line;
        static const std::unordered_map<std::string, UniformType> typeMap = {
            {"int", UniformType::Int}, {"float", UniformType::Float},
            {"bool", UniformType::Bool}, {"vec2", UniformType::Vec2},
            {"vec3", UniformType::Vec3}, {"vec4", UniformType::Vec4},
        };
        while (std::getline(ss, line)) {
            std::smatch sm;
            if (std::regex_search(line, sm, scalarDecl)) {
                ScalarUniformInfo u;
                u.type = typeMap.at(sm[1].str());
                u.name = sm[2].str();
                uniforms.push_back(u);
            }
        }
    }
};

// Maps the format token parsed out of `layout(rg8, binding=0) uniform image2D`
// to raylib's pixel format enum. Extend as you use more formats.
inline int ToRLFormat(const std::string& glslFormat) {
    static const std::unordered_map<std::string, int> table = {
        {"r8",     RL_PIXELFORMAT_UNCOMPRESSED_GRAYSCALE},
        {"rg8",    RL_PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA},
        {"rgba8",  RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8},
        {"r32f",   RL_PIXELFORMAT_UNCOMPRESSED_R32},
        {"rgba32f",RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32},
    };
    auto it = table.find(glslFormat);
    if (it == table.end())
        throw std::runtime_error("ToRLFormat: unmapped GLSL format '" + glslFormat + "'");
    return it->second;
}


// ------------------------------------------------------------------
// RenderGraph class
// ------------------------------------------------------------------

class RenderGraph {
public:
    RenderGraph() = default;
    ~RenderGraph();

    RenderGraph(const RenderGraph&) = delete;
    RenderGraph& operator=(const RenderGraph&) = delete;
    RenderGraph(RenderGraph&& other) noexcept;
    RenderGraph& operator=(RenderGraph&& other) noexcept;

    bool Initialize(int width, int height);
    void Unload();

    ShaderHandle CreateShader(const std::string& name, const std::string& fragmentShaderPath,
                   const std::string& vertexShaderPath = "");
    ShaderHandle CreateDefaultShader(const std::string& name);
    bool DestroyShader(const std::string& name);

    PassHandle CreatePass(const std::string& name,
                                const std::string& shaderName,
                                bool persistentOutput = false);
    PassHandle CreateTextureCopyPass(const std::string& name,
                                  float resolutionScale = 1.0f,
                                  TextureFilter filter = TEXTURE_FILTER_BILINEAR);

    PassHandle CreatePassFromFile(const std::string& name,
                        const std::string& fragmentShaderPath,
                        const std::string& vertexShaderPath = "",
                        bool persistentOutput = false);
    bool DestroyPass(const std::string& name);
    void Resize(int width, int height);
    void BeginScene(Color clearColor = BLANK);
    void BeginScene(const std::string& sceneName, Color clearColor = BLANK);
    void EndScene();

    bool CreateSceneTexture(const std::string& sourceName, int width, int height);
    bool DestroySceneTexture(const std::string& sourceName);
    RenderTexture2D* GetNamedSceneTexture(const std::string& sourceName);
    const RenderTexture2D* GetNamedSceneTexture(const std::string& sourceName) const;
    void Apply();
    void DrawOutput(Rectangle destination, Color tint = WHITE, int blendMode = -1) const;

    ProcessPass* FindPass(const std::string& name);
    const ProcessPass* FindPass(const std::string& name) const;

    bool SetEnabled(const std::string& passName, bool enabled);
    bool SetPassClass(const std::string& passName, const std::string& className);
    bool SetPassTargetFramebuffer(const std::string& passName, const std::string& framebufferName);
    bool SetPassOutputFormat(const std::string& passName, int format, bool withDepth = true);
    
    bool SetPersistentOutput(const std::string& passName, bool persistent);
    bool SetResolutionScale(const std::string& passName, float scale);
    bool SetOutputFilter(const std::string& passName, TextureFilter filter);
    bool SetFeedbackEnabled(const std::string& passName, bool enabled);

    bool SetPassHistory(const std::string& passName, bool enable);
    const Texture2D* GetPassHistoryTexture(const std::string& passName) const;
    static constexpr const char* HistorySourceName() { return "__history__"; }

    bool SetPassViewport(const std::string& passName, Rectangle rect);
    bool SetPassScissor(const std::string& passName, Rectangle rect);
    bool SetComputeViewport(const std::string& passName, Rectangle rect);
    bool SetComputeScissor(const std::string& passName, Rectangle rect);
    bool SetGeometryViewport(const std::string& passName, Rectangle rect);
    bool SetGeometryScissor(const std::string& passName, Rectangle rect);

    bool SetPassGenerateMipmaps(const std::string& passName, bool enable);

    // Render state (blend/depth/stencil) for ProcessPass and GeometryPass.
    // Replaces the previously-hardcoded rlDisableColorBlend()-only behavior;
    // depth/stencil test default to off (matching prior behavior, where a
    // pass relied on whatever state user code had set before Apply()).
    bool SetPassBlendMode(const std::string& passName, PassBlendMode mode, int customBlendMode = BLEND_ALPHA);
    bool SetPassDepthTest(const std::string& passName, bool enabled, bool writeEnabled = true);
    bool SetPassStencilTest(const std::string& passName, bool enabled, int writeMask = 0xFF);

    bool SetGeometryBlendMode(const std::string& passName, PassBlendMode mode, int customBlendMode = BLEND_ALPHA);
    bool SetGeometryDepthTest(const std::string& passName, bool enabled, bool writeEnabled = true);

    // Texture binding API for GeometryPass -- mirrors CreateTextureBinding/
    // SetTexture/SetTexture3D above (ProcessPass), but there's no
    // SetGeometryTextureSource equivalent yet: only a directly-supplied
    // Texture2D (2D or raw 3D id) is supported, not "read from another
    // pass's output" the way ProcessPass bindings can.
    GeometryTextureBindingHandle CreateGeometryTextureBinding(const std::string& passName, const std::string& bindingName, const std::string& uniformName);
    bool SetGeometryTexture(const std::string& passName, const std::string& bindingName, Texture2D texture, const std::string& texturePath = "");
    bool SetGeometryTexture3D(const std::string& passName, const std::string& bindingName, Texture2D texture, const std::string& texturePath = "");
    bool DestroyGeometryTextureBinding(const std::string& passName, const std::string& bindingName);

    bool SetPassPriority(const std::string& passName, int priority);
    bool SetComputePriority(const std::string& passName, int priority);
    bool SetGeometryPriority(const std::string& passName, int priority);

    bool SetPassCondition(const std::string& passName, std::function<bool()> condition);
    bool SetGroupCondition(const std::string& groupName, std::function<bool()> condition);
    bool SetComputePassCondition(const std::string& passName, std::function<bool()> condition);
    bool SetGeometryPassCondition(const std::string& passName, std::function<bool()> condition);

    bool SetComputePersistentState(const std::string& passName, bool enable);

    const std::string& GetLastError() const { return lastError_; }
    void ClearError() { lastError_.clear(); }

    void SetProfilingEnabled(bool enabled);
    bool IsProfilingEnabled() const;
    float GetGPUTime(const std::string& name) const;  // milliseconds
    const std::unordered_map<std::string, float>& GetGPUTimeMap() const;

    bool ExportGraphToDot(const std::string& filePath) const;
    bool ImportDependencies(const std::string& filePath);

    // Uniform setters for post-process passes
    bool SetFloat(const std::string& passName, const std::string& uniformName, float value);
    bool SetInt(const std::string& passName, const std::string& uniformName, int value);
    bool SetVector2(const std::string& passName, const std::string& uniformName, Vector2 value);
    bool SetVector3(const std::string& passName, const std::string& uniformName, Vector3 value);
    bool SetVector4(const std::string& passName, const std::string& uniformName, Vector4 value);
    bool SetColor(const std::string& passName, const std::string& uniformName, Color value);
    
    // Uniform setters for compute passes
    bool SetComputeFloat(const std::string& passName, const std::string& uniformName, float value);
    bool SetComputeInt(const std::string& passName, const std::string& uniformName, int value);
    bool SetComputeVector2(const std::string& passName, const std::string& uniformName, Vector2 value);
    bool SetComputeVector3(const std::string& passName, const std::string& uniformName, Vector3 value);
    bool SetComputeVector4(const std::string& passName, const std::string& uniformName, Vector4 value);
    bool SetComputeColor(const std::string& passName, const std::string& uniformName, Color value);

    // Uniform setters for geometry passes
    bool SetGeometryFloat(const std::string& passName, const std::string& uniformName, float value);
    bool SetGeometryInt(const std::string& passName, const std::string& uniformName, int value);
    bool SetGeometryVector2(const std::string& passName, const std::string& uniformName, Vector2 value);
    bool SetGeometryVector3(const std::string& passName, const std::string& uniformName, Vector3 value);
    bool SetGeometryVector4(const std::string& passName, const std::string& uniformName, Vector4 value);
    bool SetGeometryColor(const std::string& passName, const std::string& uniformName, Color value);
    bool SetGeometryMatrix(const std::string& passName, const std::string& uniformName, const Matrix& value);

    // Attributeless (SSBO/gl_VertexID-driven) draw support -- see
    // GeometryPass::attributelessDraw/drawItemsProvider for what these
    // configure. SetGeometryAttributelessDraw is a one-time setup call
    // (which primitive, VAO-bound or not); SetGeometryDrawItemsProvider
    // is what actually drives per-frame draw calls -- call it even for a
    // pass that already binds a real vertex buffer in drawCallback if you
    // just want the graph to issue multiple SSBO-bound draw calls for you.
    bool SetGeometryAttributelessDraw(const std::string& passName, PrimitiveMode mode);
    bool SetGeometryDrawItemsProvider(const std::string& passName, GeometryDrawItemsProvider provider);

    bool BindUniformBufferToGeometryPass(const std::string& passName, const std::string& bufferName, unsigned int bindingIndex);
    bool UnbindUniformBufferFromGeometryPass(const std::string& passName, unsigned int bindingIndex);

    // Class-based uniform setters
    bool SetClassFloat(const std::string& className, const std::string& uniformName, float value);
    bool SetClassInt(const std::string& className, const std::string& uniformName, int value);
    bool SetClassVector2(const std::string& className, const std::string& uniformName, Vector2 value);
    bool SetClassVector3(const std::string& className, const std::string& uniformName, Vector3 value);
    bool SetClassVector4(const std::string& className, const std::string& uniformName, Vector4 value);
    bool SetClassColor(const std::string& className, const std::string& uniformName, Color value);

    // Texture binding for post-process passes
    PassTextureBindingHandle CreateTextureBinding(const std::string& passName, const std::string& bindingName, const std::string& uniformName);
    bool SetTexture(const std::string& passName, const std::string& bindingName, Texture2D texture, const std::string& texturePath = "");
    // For a binding declared as sampler3D in the shader. `texture` must
    // wrap a texture id you created yourself as GL_TEXTURE_3D (raw
    // glGenTextures + glTexImage3D -- there's no rlLoadTexture equivalent
    // for volume textures, same reason CreateTextureArrayFramebuffer goes
    // through raw GL). Only texture.id is actually used for binding;
    // width/height/mipmaps/format on the Texture2D you pass in are not
    // interpreted. Clears isCubemap/isTextureArray on the binding, same
    // way SetTexture doesn't touch this binding's "kind" flags except to
    // make sure only one of the three is ever set.
    bool SetTexture3D(const std::string& passName, const std::string& bindingName, Texture2D texture, const std::string& texturePath = "");
    // Loads an image laid out as a tilesX*tilesY grid of equal-sized tiles
    // (e.g. a "foo_2x2.png" dither atlas) and uploads it as a genuine
    // GL_TEXTURE_3D with tilesX*tilesY layers -- the id you then hand to
    // SetTexture3D/SetGeometryTexture3D. Same reasoning as
    // CreateTextureArrayFramebuffer: no rlLoadTexture equivalent for
    // volume textures, so this goes through raw GL via PixelFormatToGL
    // (render_graph_impl.cpp); only RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    // is supported for now, since that's what ImageFormat() normalizes the
    // atlas to before deinterleaving. Tiles are read top-down (row-major,
    // matching how the file displays) but written bottom-up into layer
    // index (the visual BOTTOM-most tile becomes layer 0, the visual TOP
    // tile becomes the last layer) -- matches Dither3D_2x2.png's
    // convention, where the sparsest/1-dot tile is at the bottom and the
    // densest tile is at the top. If you load an atlas authored the other
    // way, flip this. Returns a zero-id Texture2D on failure (bad path,
    // dimensions not divisible by tilesX/tilesY, etc); width/height/format
    // on the returned Texture2D describe one tile/layer, for reference
    // only -- same as SetTexture3D, only .id is read when binding it.
    Texture2D LoadTexture3DFromAtlas(const std::string& path, int tilesX, int tilesY);
    bool SetTextureSource(const std::string& passName, const std::string& bindingName, const std::string& sourceName);
    // Like SetTextureSource, but for reaching a color attachment other
    // than index 0 of a multi-attachment Framebuffer (e.g. the "normal"
    // channel of a G-buffer that also has "albedo" at index 0). Works for
    // both ordinary and cubemap framebuffers; sets binding.isCubemap
    // automatically when framebufferName was created with
    // CreateCubemapFramebuffer.
    bool SetTextureSourceFramebufferAttachment(const std::string& passName, const std::string& bindingName, const std::string& framebufferName, std::size_t attachmentIndex);
    bool DestroyTextureBinding(const std::string& passName, const std::string& bindingName);
    bool ClearTextureBindings(const std::string& passName);

    // Compute image binding API
    ComputeImageBindingHandle CreateComputeImageBinding(const std::string& passName, const std::string& bindingName, const std::string& uniformName, bool readOnly = true, int format = RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, int imageUnit = -1);
    bool SetComputeImageTexture(const std::string& passName, const std::string& bindingName, Texture2D texture);
    bool SetComputeImageSource(const std::string& passName, const std::string& bindingName, const std::string& sourceName);
    // Wires a binding to RenderGraph's kernel *image* registry instead of
    // a concrete texture or pass source -- the texture is looked up under
    // `registryName` in the registry fresh every dispatch, so it works
    // regardless of whether RegisterImage(registryName, ...) was called
    // before or after this. This is what CreateComputePassFromKernel uses
    // internally; call it directly only if you're wiring up a compute
    // image binding by hand and want the same order-independent behavior.
    bool SetComputeImageRegistrySource(const std::string& passName, const std::string& bindingName, const std::string& registryName);
    bool DestroyComputeImageBinding(const std::string& passName, const std::string& bindingName);
    bool ClearComputeImageBindings(const std::string& passName);

    // Compute sampler binding
    ComputeTextureBindingHandle CreateComputeTextureBinding(const std::string& passName, const std::string& bindingName, const std::string& uniformName);
    bool SetComputeTexture(const std::string& passName, const std::string& bindingName, Texture2D texture);
    bool SetComputeTextureSource(const std::string& passName, const std::string& bindingName, const std::string& sourceName);
    // Same as SetComputeImageRegistrySource, resolved against the kernel
    // *sampler* registry (RegisterSampler) instead.
    bool SetComputeTextureRegistrySource(const std::string& passName, const std::string& bindingName, const std::string& registryName);
    bool DestroyComputeTextureBinding(const std::string& passName, const std::string& bindingName);
    bool ClearComputeTextureBindings(const std::string& passName);
    bool SetComputeImageRawTexture(const std::string& passName, const std::string& bindingName, unsigned int textureId, unsigned int glInternalFormat);

    // Binding-copy API: recreates fromPass's bindings on toPass (fresh
    // CreateComputeImageBinding/CreateComputeTextureBinding calls against
    // toPass's own program, so locations and image units are toPass's own
    // -- not a shallow struct copy). Whatever fromPass's binding was
    // pointed at (a literal texture, or a source pass/group/framebuffer
    // name) carries over. Existing bindings on toPass with a colliding
    // name/uniform are left alone and that one binding is skipped -- the
    // rest still copy. Meant for "new pass needs the same resources an
    // existing one already has wired up" (ping-pong variants, a second
    // pass over the same field) instead of re-deriving each binding by
    // hand.
    bool CopyComputeImageBindings(const std::string& fromPassName, const std::string& toPassName);
    bool CopyComputeTextureBindings(const std::string& fromPassName, const std::string& toPassName);
    bool CopyComputeBufferBindings(const std::string& fromPassName, const std::string& toPassName);
    // Convenience: all three plus imageBindings/textureBindings/buffer
    // bindings' matching uniform values (anything already in fromPass's
    // Uniform cache, pushed through the normal Set* path so toPass's own
    // dirty-check still applies).
    bool CopyAllComputeBindings(const std::string& fromPassName, const std::string& toPassName);

    // Post-process groups
    GroupHandle CreateProcessGroup(const std::string& name);
    bool DestroyProcessGroup(const std::string& name);
    bool AttachPassToGroup(const std::string& groupName, const std::string& passName);
    bool RemovePassFromGroup(const std::string& groupName, const std::string& passName);
    bool SetGroupEnabled(const std::string& groupName, bool enabled);
    bool SetGroupSource(const std::string& groupName, const std::string& sourceName);
    ProcessGroup* FindGroup(const std::string& name);
    const ProcessGroup* FindGroup(const std::string& name) const;

    // Group templates
    bool CreateGroupTemplate(const std::string& name);
    bool DestroyGroupTemplate(const std::string& name);
    bool AttachPassToGroupTemplate(const std::string& templateName,
                                const std::string& passName,
                                const std::string& shaderName);
    // Returns a GroupHandle for the ProcessGroup created under instanceName
    // (an instance's group always shares its name), not a separate
    // "instance" handle -- use FindGroupInstance(name) if you need the
    // ProcessGroupInstance record itself.
    GroupHandle CreateGroupInstance(const std::string& instanceName,
                                 const std::string& templateName);
    bool DestroyGroupInstance(const std::string& instanceName);
    ProcessGroupTemplate* FindGroupTemplate(const std::string& name);
    const ProcessGroupTemplate* FindGroupTemplate(const std::string& name) const;
    ProcessGroupInstance* FindGroupInstance(const std::string& name);
    const ProcessGroupInstance* FindGroupInstance(const std::string& name) const;

    const Texture2D* GetGroupOutputTexture(const std::string& groupName) const;
    static constexpr const char* SceneSourceName() { return "__scene__"; }

    // Shader reloading
    bool ReloadPassShader(const std::string& passName);
    bool ReloadShader(const std::string& shaderName);
    bool ReloadComputeShader(const std::string& shaderName);

    // Accessors
    std::vector<ProcessShader>& GetShaders();
    const std::vector<ProcessShader>& GetShaders() const;
    std::vector<ProcessPass>& GetPasses();
    const std::vector<ProcessPass>& GetPasses() const;
    std::vector<ProcessGroup>& GetGroups();
    const std::vector<ProcessGroup>& GetGroups() const;
    std::vector<ProcessGroupTemplate>& GetGroupTemplates();
    const std::vector<ProcessGroupTemplate>& GetGroupTemplates() const;
    std::vector<ProcessGroupInstance>& GetGroupInstances();
    const std::vector<ProcessGroupInstance>& GetGroupInstances() const;

    // Manual reordering
    // Takes the pass by name (like every other mutator here) rather than an
    // index into passes_ -- an index is exactly the kind of handle the
    // Handle<Tag> comment above warns about: it silently goes stale (or
    // silently starts referring to a *different* pass) the moment anything
    // earlier in passes_ is destroyed. A PassHandle overload is provided
    // below alongside the other handle-typed convenience overloads.
    bool MovePassUp(const std::string& passName);
    bool MovePassDown(const std::string& passName);

    // Preview
    void SetPreviewAllPassesEnabled(bool enable);
    bool GetPreviewAllPasses() const;
    const Texture2D* GetPassOutputTexture(const std::string& passName) const;

    // Output access
    const Texture2D& GetOutputTexture() const;
    const RenderTexture2D& GetSceneTexture() const;
    int GetWidth() const;
    int GetHeight() const;

    // Framebuffer management
    FramebufferHandle CreateFramebuffer(const std::string& name, int width, int height, const std::vector<int>& colorFormats, bool withDepth = true, bool depthAsTexture = false);
    // Same as CreateFramebuffer, but every color attachment is a
    // GL_TEXTURE_CUBE_MAP (all 6 faces allocated empty, via
    // rlLoadTextureCubemap) instead of a 2D texture, and width==height==
    // faceSize. Depth is a single faceSize x faceSize renderbuffer shared
    // (re-cleared) across all 6 faces -- it's only used for depth testing
    // during the capture, not read back afterward. Pass the resulting
    // handle to SetGeometryPassTargetFramebuffer on a pass that also has
    // SetGeometryPassCubemapCapture set.
    FramebufferHandle CreateCubemapFramebuffer(const std::string& name, int faceSize, const std::vector<int>& colorFormats, bool withDepth = true);
    // Allocates `layers` independent width x height images per color
    // format, as one GL_TEXTURE_2D_ARRAY per format (not one texture per
    // layer). Bypasses rlgl for the texture itself -- rlLoadTexture has
    // no array-texture equivalent -- so colorFormats is limited to
    // whatever PixelFormatToGL (render_graph_impl.cpp) maps; extend that
    // table before passing a format it doesn't recognize.
    FramebufferHandle CreateTextureArrayFramebuffer(const std::string& name, int width, int height, int layers, const std::vector<int>& colorFormats, bool withDepth = true);
    bool DestroyFramebuffer(const std::string& name);
    bool ResizeFramebuffer(const std::string& name, int width, int height);
    Framebuffer* FindFramebuffer(const std::string& name);
    const Framebuffer* FindFramebuffer(const std::string& name) const;
    bool BlitFramebuffer(const std::string& srcName, Rectangle srcRect, const std::string& dstName, Rectangle dstRect, int mask);
    void BindFramebuffer(const std::string& name);
    void UnbindFramebuffer();

    // Compute shader / buffer / pass
    ComputeShaderHandle CreateComputeShader(const std::string& name, const std::string& computeShaderPath);
    // Same as CreateComputeShader, but compiles from an in-memory GLSL
    // string instead of reading a file from disk -- for shader text
    // assembled at runtime (e.g. ShaderKernelLibrary's per-kernel
    // fullSource). Shares the same name registry as CreateComputeShader:
    // 'name' must be unique across both, and shaders created either way
    // are destroyed the same way (DestroyComputeShader). ReloadComputeShader
    // will fail (returns false) for a shader created here, since there's
    // no backing file path to re-read from.
    ComputeShaderHandle CreateComputeShaderFromSource(const std::string& name, const std::string& source);
    bool DestroyComputeShader(const std::string& name);
    ComputeProgram* FindComputeProgram(const std::string& name);
    const ComputeProgram* FindComputeProgram(const std::string& name) const;

    BufferHandle CreateBuffer(const std::string& name, unsigned int sizeBytes,
                          const void* initialData = nullptr,
                          int usageHint = RL_DYNAMIC_COPY);
    bool DestroyBuffer(const std::string& name);
    // Reallocates the SSBO at a new size, replacing its GL object in place.
    // Existing bindings reference buffers by name (not by pointer), so no
    // pass rebinding is needed afterward. When preserveData is true (the
    // default), the overlapping min(oldSize, newSize) bytes are copied
    // into the new buffer; pass false to skip that read-back if the
    // contents are about to be fully overwritten anyway.
    bool ResizeBuffer(const std::string& name, unsigned int newSizeBytes, bool preserveData = true);
    bool UpdateBuffer(const std::string& name, const void* data, unsigned int sizeBytes, unsigned int offset = 0);
    bool ReadBuffer(const std::string& name, void* outData, unsigned int sizeBytes, unsigned int offset = 0);
    ShaderBuffer* FindBuffer(const std::string& name);
    const ShaderBuffer* FindBuffer(const std::string& name) const;

    ComputePassHandle CreateComputePass(const std::string& name, const std::string& computeShaderName);
    bool DestroyComputePass(const std::string& name);
    bool BindBufferToComputePass(const std::string& passName, const std::string& bufferName, unsigned int bindingIndex);
    bool SwapComputePassBufferBindings(const std::string& passName, unsigned int bindingIndexA, unsigned int bindingIndexB);
    bool UnbindBufferFromComputePass(const std::string& passName, unsigned int bindingIndex);
    bool SetComputeWorkgroupSize(const std::string& passName, unsigned int x, unsigned int y, unsigned int z);
    bool SetComputePassEnabled(const std::string& passName, bool enabled);
    bool AttachComputePassToGraph(const std::string& passName);
    bool RemoveComputePassFromGraph(const std::string& passName);
    bool InsertComputePassBefore(const std::string& computePassName, const std::string& beforeNodeName);
    bool InsertComputePassAfter(const std::string& computePassName, const std::string& afterNodeName);
    bool DispatchComputePass(const std::string& passName);

    // ------------------------------------------------------------------
    // Kernel shaders: HLSL-style multi-kernel GLSL, or ordinary GLSL --
    // auto-detected on load (see ShaderKernelLibrary above).
    // ------------------------------------------------------------------

    // Parses `path` (or raw `source`) into one or more KernelInfo entries.
    // Safe to call more than once; later kernels with a name collision
    // replace earlier ones. Throws std::runtime_error on a malformed file.
    void LoadKernelFile(const std::string& path, const std::string& plainKernelName = "");
    void LoadKernelSource(const std::string& source, const std::string& plainKernelName = "");
    const KernelInfo* FindKernel(const std::string& name) const;
    std::vector<std::string> KernelNames() const;

    // Registers a texture/buffer under `name` for CreateComputePassFromKernel
    // to pull from automatically -- this *is* the resource registry; there
    // is no separate object to construct or pass around.
    //
    // Order-independent: CreateComputePassFromKernel wires bindings to the
    // registry *by name*, resolved fresh every dispatch (see
    // SetComputeImageRegistrySource/SetComputeTextureRegistrySource) --
    // it does not require the name to already be registered when the pass
    // is created. Call Register*/CreateComputePassFromKernel in whichever
    // order is convenient; as long as the name is registered by the time
    // Apply() actually dispatches that pass, it resolves correctly. Also
    // means re-registering a name later (swapping in a different texture)
    // takes effect on the very next dispatch, without touching the pass.
    // To bypass the registry for one binding and pin a concrete texture/
    // buffer instead, use the pass-level Set*/BindBufferToComputePass
    // calls directly, same as any other compute pass -- those clear the
    // registry link for that binding.
    //
    // Images and samplers are separate registries, matching the separate
    // `image2D` and `sampler2D` reflection lists on KernelInfo -- a kernel
    // that declares `trailMap` as `image2D` looks it up in the image
    // registry, one that declares it as `sampler2D` looks it up in the
    // sampler registry, so the same name can resolve to different
    // textures depending on which way a given kernel uses it (e.g. a
    // ping-pong buffer read as a sampler in one kernel and written as an
    // image in another).
    void RegisterImage(const std::string& name, Texture2D texture);
    void RegisterSampler(const std::string& name, Texture2D texture);
    // Convenience for a texture used the same way everywhere: registers
    // `name` under both the image and sampler registries in one call.
    // Equivalent to calling RegisterImage(name, texture) followed by
    // RegisterSampler(name, texture).
    void RegisterTexture(const std::string& name, Texture2D texture);
    void RegisterBuffer(const std::string& name, const BufferHandle& buffer);

    // One-call kernel setup: compiles the kernel, creates its pass, wires
    // every image/texture/buffer the kernel's reflection says it needs to
    // the registry above by name (see RegisterImage/RegisterSampler/
    // RegisterBuffer -- registration can happen before or after this
    // call), sets the workgroup size from the kernel's own [numthreads]
    // (or its own layout(local_size_*) for a plain GLSL kernel), and
    // attaches the pass to the graph. Image and buffer bindings are made
    // at the exact binding number the shader declares (`layout(...,
    // binding = N)`), or the next free slot if the shader left it
    // unspecified -- reflected/assigned automatically, nothing to specify
    // by hand. Also registers the kernel's uniform names against the
    // resulting pass so SetUniform() below finds it automatically.
    //
    // itemsX/Y/Z are total work-item counts along each axis (e.g.
    // numAgents,1,1 for a per-agent kernel, or width,height,1 for a
    // per-pixel kernel) -- NOT group counts.
    //
    // This is a one-time setup call per kernel, like the graph's other
    // Create* calls: every RenderGraph entity this makes fails (returns
    // an invalid handle) if that name already exists, so calling it twice
    // for the same kernel is a bug, not a no-op refresh. Throws
    // std::runtime_error immediately on any setup failure (duplicate
    // kernel name, unresolved uniform location, unmapped GLSL format,
    // etc.) rather than leaving a silently half-bound pass in the graph.
    // Registry entries that are still missing at this point are NOT a
    // setup failure -- run Validate() after your registration calls to
    // catch a name that never got registered at all.
    ComputePassHandle CreateComputePassFromKernel(const std::string& kernelName,
                                                   unsigned int itemsX,
                                                   unsigned int itemsY = 1,
                                                   unsigned int itemsZ = 1);

    // ------------------------------------------------------------------
    // Automatic uniforms
    // ------------------------------------------------------------------
    // Set a named uniform once here and it's pushed to every kernel pass
    // (created via CreateComputePassFromKernel) whose kernel reflection
    // says it declares that name -- automatically, the next time Apply()
    // runs. There's no registry object to hold onto and no ApplyTo() to
    // remember: Apply() does that internally, and (like the graph's own
    // per-pass Set*) only pushes a (uniform, pass) pair whose value
    // actually changed since that pass last received it, so an unchanged
    // frame costs a handful of version-number comparisons, not redundant
    // GL uniform pushes. For a pass not created via CreateComputePassFromKernel
    // (i.e. no known kernel reflection), push uniforms directly with the
    // existing SetComputeFloat/SetFloat/SetGeometryFloat (etc.) calls.
    void SetUniform(const std::string& name, float value);
    void SetUniform(const std::string& name, int value);
    void SetUniform(const std::string& name, Vector2 value);
    void SetUniform(const std::string& name, Vector3 value);
    void SetUniform(const std::string& name, Vector4 value);
    void SetUniform(const std::string& name, Color value);

    ComputePass* FindComputePass(const std::string& name);
    const ComputePass* FindComputePass(const std::string& name) const;

    // Compute groups
    ComputeGroupHandle CreateComputeGroup(const std::string& name);
    bool DestroyComputeGroup(const std::string& name);
    bool AttachComputePassToGroup(const std::string& groupName, const std::string& passName);
    bool RemoveComputePassFromGroup(const std::string& groupName, const std::string& passName);
    bool SetComputeGroupEnabled(const std::string& groupName, bool enabled);
    bool SetComputeGroupCondition(const std::string& groupName, std::function<bool()> condition);
    bool SetComputeGroupPriority(const std::string& groupName, int priority);
    ComputeGroup* FindComputeGroup(const std::string& name);
    const ComputeGroup* FindComputeGroup(const std::string& name) const;
    std::vector<ComputeGroup>& GetComputeGroups();
    const std::vector<ComputeGroup>& GetComputeGroups() const;

    bool AttachComputeGroupToGraph(const std::string& groupName);
    bool RemoveComputeGroupFromGraph(const std::string& groupName);
    bool InsertComputeGroupBefore(const std::string& computeGroupName, const std::string& beforeNodeName);
    bool InsertComputeGroupAfter(const std::string& computeGroupName, const std::string& afterNodeName);

    // Class management (multiple classes per pass)
    bool AddPassClass(const std::string& passName, const std::string& className);
    bool RemovePassClass(const std::string& passName, const std::string& className);
    bool AddComputeClass(const std::string& passName, const std::string& className);
    bool RemoveComputeClass(const std::string& passName, const std::string& className);
    bool AddGeometryClass(const std::string& passName, const std::string& className);
    bool RemoveGeometryClass(const std::string& passName, const std::string& className);

    // Geometry passes
    GeometryPassHandle CreateGeometryPass(const std::string& name, const std::string& shaderName);
    bool DestroyGeometryPass(const std::string& name);
    GeometryPass* FindGeometryPass(const std::string& name);
    const GeometryPass* FindGeometryPass(const std::string& name) const;
    bool AttachGeometryPassToGraph(const std::string& passName);
    bool RemoveGeometryPassFromGraph(const std::string& passName);
    bool InsertGeometryPassBefore(const std::string& passName, const std::string& beforeNodeName);
    bool InsertGeometryPassAfter(const std::string& passName, const std::string& afterNodeName);
    bool SetGeometryPassTargetFramebuffer(const std::string& passName, const std::string& framebufferName);
    bool SetGeometryPassCallback(const std::string& passName, std::function<void()> callback);
    // Turns passName into a 6-face cubemap capture (see
    // GeometryPass::isCubemapCapture). camera must outlive the pass --
    // the graph mutates its position/target/up/fovy in place before each
    // of the 6 drawCallback invocations; the caller's drawCallback keeps
    // reading it via BeginMode3D(*camera) exactly like a normal geometry
    // pass would. Pass nullptr to turn capture mode back off.
    bool SetGeometryPassCubemapCapture(const std::string& passName, Camera3D* camera, Vector3 probePosition, float nearPlane = 0.05f, float farPlane = 100.0f);
    // Sets which array layer a pass targeting a texture-array Framebuffer
    // writes into on its next Apply(). No-op (returns false) if passName
    // doesn't exist; silently ignored at draw time if its target
    // framebuffer isn't a texture array.
    bool SetGeometryPassTargetArrayLayer(const std::string& passName, int layer);
    bool SetGeometryPassEnabled(const std::string& passName, bool enabled);
    bool BindBufferToGeometryPass(const std::string& passName, const std::string& bufferName, unsigned int bindingIndex);
    bool UnbindBufferFromGeometryPass(const std::string& passName, unsigned int bindingIndex);
    bool SwapGeometryPassBufferBindings(const std::string& passName, unsigned int bindingIndexA, unsigned int bindingIndexB);

    bool ValidateGraph() const;

    // Order-independent integrity check: every named reference on every
    // entity (texture/image sources, SSBO/uniform-buffer bindings, target
    // framebuffers, shader/program resolution, group membership, and
    // AddDependency() entries), checked against what actually exists.
    // Returns one issue per broken reference; an empty result means the
    // graph is fully self-consistent. Cheap enough to run once after
    // building the graph and log the result, rather than discovering
    // typos one silently-no-op'd pass at a time.
    std::vector<ValidationIssue> Validate() const;

    std::vector<ComputeProgram>& GetComputePrograms();
    const std::vector<ComputeProgram>& GetComputePrograms() const;
    std::vector<ComputePass>& GetComputePasses();
    const std::vector<ComputePass>& GetComputePasses() const;
    std::vector<ShaderBuffer>& GetBuffers();
    const std::vector<ShaderBuffer>& GetBuffers() const;
    ProcessShader* FindShader(const std::string& name);
    const ProcessShader* FindShader(const std::string& name) const;

    // Dependency graph
    bool AddDependency(const std::string& sourceName, const std::string& targetName);
    bool RemoveDependency(const std::string& sourceName, const std::string& targetName);
    void SetAutomaticOrdering(bool enable);
    bool GetAutomaticOrdering() const;

    // Uniform buffers
    UniformBufferHandle CreateUniformBuffer(const std::string& name, unsigned int sizeBytes, const void* initialData = nullptr, int usageHint = RL_DYNAMIC_DRAW);
    bool DestroyUniformBuffer(const std::string& name);
    bool UpdateUniformBuffer(const std::string& name, const void* data, unsigned int sizeBytes, unsigned int offset = 0);
    UniformBuffer* FindUniformBuffer(const std::string& name);
    const UniformBuffer* FindUniformBuffer(const std::string& name) const;

    bool BindUniformBufferToPass(const std::string& passName, const std::string& bufferName, unsigned int bindingIndex);
    bool UnbindUniformBufferFromPass(const std::string& passName, unsigned int bindingIndex);

    bool BindUniformBufferToComputePass(const std::string& passName, const std::string& bufferName, unsigned int bindingIndex);
    bool UnbindUniformBufferFromComputePass(const std::string& passName, unsigned int bindingIndex);

    // ------------------------------------------------------------------
    // Handle accessors
    // ------------------------------------------------------------------
    PassHandle GetPassHandle(const std::string& name) const {
        return FindPass(name) ? PassHandle(name) : PassHandle();
    }
    GeometryPassHandle GetGeometryPassHandle(const std::string& name) const {
        return FindGeometryPass(name) ? GeometryPassHandle(name) : GeometryPassHandle();
    }
    ComputePassHandle GetComputePassHandle(const std::string& name) const {
        return FindComputePass(name) ? ComputePassHandle(name) : ComputePassHandle();
    }
    ShaderHandle GetShaderHandle(const std::string& name) const {
        return FindShader(name) ? ShaderHandle(name) : ShaderHandle();
    }
    ComputeShaderHandle GetComputeShaderHandle(const std::string& name) const {
        return FindComputeProgram(name) ? ComputeShaderHandle(name) : ComputeShaderHandle();
    }
    BufferHandle GetBufferHandle(const std::string& name) const {
        return FindBuffer(name) ? BufferHandle(name) : BufferHandle();
    }
    UniformBufferHandle GetUniformBufferHandle(const std::string& name) const {
        return FindUniformBuffer(name) ? UniformBufferHandle(name) : UniformBufferHandle();
    }
    FramebufferHandle GetFramebufferHandle(const std::string& name) const {
        return FindFramebuffer(name) ? FramebufferHandle(name) : FramebufferHandle();
    }
    GroupHandle GetGroupHandle(const std::string& name) const {
        return FindGroup(name) ? GroupHandle(name) : GroupHandle();
    }
    ComputeGroupHandle GetComputeGroupHandle(const std::string& name) const {
        return FindComputeGroup(name) ? ComputeGroupHandle(name) : ComputeGroupHandle();
    }
    // Resolves a name to a TextureHandle if it corresponds to any existing
    // texture source: a pass output, a scene texture, or a framebuffer
    // color attachment. Returns an invalid handle otherwise.
    TextureHandle GetTextureHandle(const std::string& name) const {
        if (FindPass(name) || FindFramebuffer(name) || sceneTextures_.find(name) != sceneTextures_.end()) {
            return TextureHandle(name);
        }
        return TextureHandle();
    }

    // Scoped binding handle accessors
    PassTextureBindingHandle GetTextureBindingHandle(const PassHandle& pass, const std::string& bindingName) const {
        const ProcessPass* p = FindPass(pass);
        if (p) {
            for (const auto& tb : p->textureBindings) {
                if (tb.name == bindingName) return PassTextureBindingHandle(pass, bindingName);
            }
        }
        return PassTextureBindingHandle();
    }
    ComputeImageBindingHandle GetComputeImageBindingHandle(const ComputePassHandle& pass, const std::string& bindingName) const {
        const ComputePass* cp = FindComputePass(pass);
        if (cp) {
            for (const auto& ib : cp->imageBindings) {
                if (ib.name == bindingName) return ComputeImageBindingHandle(pass, bindingName);
            }
        }
        return ComputeImageBindingHandle();
    }
    ComputeTextureBindingHandle GetComputeTextureBindingHandle(const ComputePassHandle& pass, const std::string& bindingName) const {
        const ComputePass* cp = FindComputePass(pass);
        if (cp) {
            for (const auto& tb : cp->textureBindings) {
                if (tb.name == bindingName) return ComputeTextureBindingHandle(pass, bindingName);
            }
        }
        return ComputeTextureBindingHandle();
    }

    // ------------------------------------------------------------------
    // Handle-typed convenience overloads
    // ------------------------------------------------------------------
    // -- Post-process pass (PassHandle) --
    bool SetEnabled(const PassHandle& pass, bool enabled) { return SetEnabled(pass.name, enabled); }
    bool SetPassClass(const PassHandle& pass, const std::string& className) { return SetPassClass(pass.name, className); }
    bool SetPassTargetFramebuffer(const PassHandle& pass, const FramebufferHandle& fb) { return SetPassTargetFramebuffer(pass.name, fb.name); }
    bool SetPassOutputFormat(const PassHandle& pass, int format, bool withDepth = true) { return SetPassOutputFormat(pass.name, format, withDepth); }
    bool SetPassPersistentOutput(const PassHandle& pass, bool persistent) { return SetPersistentOutput(pass.name, persistent); }
    bool SetResolutionScale(const PassHandle& pass, float scale) { return SetResolutionScale(pass.name, scale); }
    bool SetOutputFilter(const PassHandle& pass, TextureFilter filter) { return SetOutputFilter(pass.name, filter); }
    bool SetFeedbackEnabled(const PassHandle& pass, bool enabled) { return SetFeedbackEnabled(pass.name, enabled); }
    bool SetPassHistory(const PassHandle& pass, bool enable) { return SetPassHistory(pass.name, enable); }
    bool SetPassViewport(const PassHandle& pass, Rectangle rect) { return SetPassViewport(pass.name, rect); }
    bool SetPassScissor(const PassHandle& pass, Rectangle rect) { return SetPassScissor(pass.name, rect); }
    bool SetPassGenerateMipmaps(const PassHandle& pass, bool enable) { return SetPassGenerateMipmaps(pass.name, enable); }
    bool SetPassBlendMode(const PassHandle& pass, PassBlendMode mode, int customBlendMode = BLEND_ALPHA) { return SetPassBlendMode(pass.name, mode, customBlendMode); }
    bool SetPassDepthTest(const PassHandle& pass, bool enabled, bool writeEnabled = true) { return SetPassDepthTest(pass.name, enabled, writeEnabled); }
    bool SetPassStencilTest(const PassHandle& pass, bool enabled, int writeMask = 0xFF) { return SetPassStencilTest(pass.name, enabled, writeMask); }
    bool SetPassPriority(const PassHandle& pass, int priority) { return SetPassPriority(pass.name, priority); }
    bool SetPassCondition(const PassHandle& pass, std::function<bool()> condition) { return SetPassCondition(pass.name, std::move(condition)); }
    bool ReloadPassShader(const PassHandle& pass) { return ReloadPassShader(pass.name); }
    bool DestroyPass(const PassHandle& pass) { return DestroyPass(pass.name); }
    bool MovePassUp(const PassHandle& pass) { return MovePassUp(pass.name); }
    bool MovePassDown(const PassHandle& pass) { return MovePassDown(pass.name); }
    bool AddPassClass(const PassHandle& pass, const std::string& className) { return AddPassClass(pass.name, className); }
    bool RemovePassClass(const PassHandle& pass, const std::string& className) { return RemovePassClass(pass.name, className); }

    bool SetFloat(const PassHandle& pass, const std::string& uniformName, float value) { return SetFloat(pass.name, uniformName, value); }
    bool SetInt(const PassHandle& pass, const std::string& uniformName, int value) { return SetInt(pass.name, uniformName, value); }
    bool SetVector2(const PassHandle& pass, const std::string& uniformName, Vector2 value) { return SetVector2(pass.name, uniformName, value); }
    bool SetVector3(const PassHandle& pass, const std::string& uniformName, Vector3 value) { return SetVector3(pass.name, uniformName, value); }
    bool SetVector4(const PassHandle& pass, const std::string& uniformName, Vector4 value) { return SetVector4(pass.name, uniformName, value); }
    bool SetColor(const PassHandle& pass, const std::string& uniformName, Color value) { return SetColor(pass.name, uniformName, value); }

    // New scoped binding overloads for post-process pass
    PassTextureBindingHandle CreateTextureBinding(const PassHandle& pass, const std::string& bindingName, const std::string& uniformName) {
        return CreateTextureBinding(pass.name, bindingName, uniformName);
    }
    bool SetTexture(const PassTextureBindingHandle& binding, Texture2D texture, const std::string& texturePath = "") {
        return SetTexture(binding.pass.name, binding.bindingName, texture, texturePath);
    }
    bool SetTexture3D(const PassTextureBindingHandle& binding, Texture2D texture, const std::string& texturePath = "") {
        return SetTexture3D(binding.pass.name, binding.bindingName, texture, texturePath);
    }
    bool SetTextureSource(const PassTextureBindingHandle& binding, const TextureHandle& source) {
        return SetTextureSource(binding.pass.name, binding.bindingName, source.name);
    }
    bool SetTextureSource(const PassTextureBindingHandle& binding, const std::string& sourceName) {
        return SetTextureSource(binding.pass.name, binding.bindingName, sourceName);
    }
    bool DestroyTextureBinding(const PassTextureBindingHandle& binding) {
        return DestroyTextureBinding(binding.pass.name, binding.bindingName);
    }

    // Existing (deprecated? keep for backwards compatibility) string+pass overloads remain.
    bool SetTexture(const PassHandle& pass, const std::string& bindingName, Texture2D texture, const std::string& texturePath = "") {
        return SetTexture(pass.name, bindingName, texture, texturePath);
    }
    bool SetTexture3D(const PassHandle& pass, const std::string& bindingName, Texture2D texture, const std::string& texturePath = "") {
        return SetTexture3D(pass.name, bindingName, texture, texturePath);
    }
    bool SetTextureSource(const PassHandle& pass, const std::string& bindingName, const TextureHandle& source) {
        return SetTextureSource(pass.name, bindingName, source.name);
    }
    bool SetTextureSourceFramebufferAttachment(const PassHandle& pass, const std::string& bindingName, const FramebufferHandle& fb, std::size_t attachmentIndex) {
        return SetTextureSourceFramebufferAttachment(pass.name, bindingName, fb.name, attachmentIndex);
    }
    bool SetTextureSource(const PassHandle& pass, const std::string& bindingName, const std::string& sourceName) {
        return SetTextureSource(pass.name, bindingName, sourceName);
    }
    bool RemoveTextureBinding(const PassHandle& pass, const std::string& bindingName) {
        return DestroyTextureBinding(pass.name, bindingName);
    }

    bool BindUniformBufferToPass(const PassHandle& pass, const UniformBufferHandle& buffer, unsigned int bindingIndex) { return BindUniformBufferToPass(pass.name, buffer.name, bindingIndex); }
    bool UnbindUniformBufferFromPass(const PassHandle& pass, unsigned int bindingIndex) { return UnbindUniformBufferFromPass(pass.name, bindingIndex); }

    bool AttachPassToGroup(const GroupHandle& group, const PassHandle& pass) { return AttachPassToGroup(group.name, pass.name); }
    bool RemovePassFromGroup(const GroupHandle& group, const PassHandle& pass) { return RemovePassFromGroup(group.name, pass.name); }

    // -- Geometry pass (GeometryPassHandle) --
    bool SetGeometryPassEnabled(const GeometryPassHandle& pass, bool enabled) { return SetGeometryPassEnabled(pass.name, enabled); }
    bool SetGeometryPassTargetFramebuffer(const GeometryPassHandle& pass, const FramebufferHandle& fb) { return SetGeometryPassTargetFramebuffer(pass.name, fb.name); }
    bool SetGeometryPassCallback(const GeometryPassHandle& pass, std::function<void()> callback) { return SetGeometryPassCallback(pass.name, std::move(callback)); }
    bool SetGeometryPassCubemapCapture(const GeometryPassHandle& pass, Camera3D* camera, Vector3 probePosition, float nearPlane = 0.05f, float farPlane = 100.0f) {
        return SetGeometryPassCubemapCapture(pass.name, camera, probePosition, nearPlane, farPlane);
    }
    bool SetGeometryPassTargetArrayLayer(const GeometryPassHandle& pass, int layer) {
        return SetGeometryPassTargetArrayLayer(pass.name, layer);
    }
    bool SetGeometryViewport(const GeometryPassHandle& pass, Rectangle rect) { return SetGeometryViewport(pass.name, rect); }
    bool SetGeometryScissor(const GeometryPassHandle& pass, Rectangle rect) { return SetGeometryScissor(pass.name, rect); }
    bool SetGeometryPriority(const GeometryPassHandle& pass, int priority) { return SetGeometryPriority(pass.name, priority); }
    bool SetGeometryPassCondition(const GeometryPassHandle& pass, std::function<bool()> condition) { return SetGeometryPassCondition(pass.name, std::move(condition)); }
    bool DestroyGeometryPass(const GeometryPassHandle& pass) { return DestroyGeometryPass(pass.name); }
    bool AddGeometryPassToGraph(const GeometryPassHandle& pass) { return AttachGeometryPassToGraph(pass.name); }
    bool RemoveGeometryPassFromGraph(const GeometryPassHandle& pass) { return RemoveGeometryPassFromGraph(pass.name); }
    bool InsertGeometryPassBefore(const GeometryPassHandle& pass, const std::string& beforeNodeName) { return InsertGeometryPassBefore(pass.name, beforeNodeName); }
    bool InsertGeometryPassAfter(const GeometryPassHandle& pass, const std::string& afterNodeName) { return InsertGeometryPassAfter(pass.name, afterNodeName); }
    bool AddGeometryClass(const GeometryPassHandle& pass, const std::string& className) { return AddGeometryClass(pass.name, className); }
    bool RemoveGeometryClass(const GeometryPassHandle& pass, const std::string& className) { return RemoveGeometryClass(pass.name, className); }

    bool SetGeometryFloat(const GeometryPassHandle& pass, const std::string& uniformName, float value) { return SetGeometryFloat(pass.name, uniformName, value); }
    bool SetGeometryInt(const GeometryPassHandle& pass, const std::string& uniformName, int value) { return SetGeometryInt(pass.name, uniformName, value); }
    bool SetGeometryVector2(const GeometryPassHandle& pass, const std::string& uniformName, Vector2 value) { return SetGeometryVector2(pass.name, uniformName, value); }
    bool SetGeometryVector3(const GeometryPassHandle& pass, const std::string& uniformName, Vector3 value) { return SetGeometryVector3(pass.name, uniformName, value); }
    bool SetGeometryVector4(const GeometryPassHandle& pass, const std::string& uniformName, Vector4 value) { return SetGeometryVector4(pass.name, uniformName, value); }
    bool SetGeometryColor(const GeometryPassHandle& pass, const std::string& uniformName, Color value) { return SetGeometryColor(pass.name, uniformName, value); }
    bool SetGeometryMatrix(const GeometryPassHandle& pass, const std::string& uniformName, const Matrix& value) { return SetGeometryMatrix(pass.name, uniformName, value); }
    bool SetGeometryAttributelessDraw(const GeometryPassHandle& pass, PrimitiveMode mode) { return SetGeometryAttributelessDraw(pass.name, mode); }
    bool SetGeometryDrawItemsProvider(const GeometryPassHandle& pass, GeometryDrawItemsProvider provider) { return SetGeometryDrawItemsProvider(pass.name, std::move(provider)); }

    bool BindBufferToGeometryPass(const GeometryPassHandle& pass, const BufferHandle& buffer, unsigned int bindingIndex) { return BindBufferToGeometryPass(pass.name, buffer.name, bindingIndex); }
    bool UnbindBufferFromGeometryPass(const GeometryPassHandle& pass, unsigned int bindingIndex) { return UnbindBufferFromGeometryPass(pass.name, bindingIndex); }
    bool BindUniformBufferToGeometryPass(const GeometryPassHandle& pass, const UniformBufferHandle& buffer, unsigned int bindingIndex) { return BindUniformBufferToGeometryPass(pass.name, buffer.name, bindingIndex); }
    bool UnbindUniformBufferFromGeometryPass(const GeometryPassHandle& pass, unsigned int bindingIndex) { return UnbindUniformBufferFromGeometryPass(pass.name, bindingIndex); }
    bool SetGeometryBlendMode(const GeometryPassHandle& pass, PassBlendMode mode, int customBlendMode = BLEND_ALPHA) { return SetGeometryBlendMode(pass.name, mode, customBlendMode); }
    bool SetGeometryDepthTest(const GeometryPassHandle& pass, bool enabled, bool writeEnabled = true) { return SetGeometryDepthTest(pass.name, enabled, writeEnabled); }

    GeometryTextureBindingHandle CreateGeometryTextureBinding(const GeometryPassHandle& pass, const std::string& bindingName, const std::string& uniformName) {
        return CreateGeometryTextureBinding(pass.name, bindingName, uniformName);
    }
    bool SetGeometryTexture(const GeometryTextureBindingHandle& binding, Texture2D texture, const std::string& texturePath = "") {
        return SetGeometryTexture(binding.pass.name, binding.bindingName, texture, texturePath);
    }
    bool SetGeometryTexture3D(const GeometryTextureBindingHandle& binding, Texture2D texture, const std::string& texturePath = "") {
        return SetGeometryTexture3D(binding.pass.name, binding.bindingName, texture, texturePath);
    }
    bool DestroyGeometryTextureBinding(const GeometryTextureBindingHandle& binding) {
        return DestroyGeometryTextureBinding(binding.pass.name, binding.bindingName);
    }

    // -- Compute pass (ComputePassHandle) --
    bool SetComputeEnabled(const ComputePassHandle& pass, bool enabled) { return SetComputePassEnabled(pass.name, enabled); }
    bool SetComputeWorkgroupSize(const ComputePassHandle& pass, unsigned int x, unsigned int y, unsigned int z) { return SetComputeWorkgroupSize(pass.name, x, y, z); }
    bool SetComputeViewport(const ComputePassHandle& pass, Rectangle rect) { return SetComputeViewport(pass.name, rect); }
    bool SetComputeScissor(const ComputePassHandle& pass, Rectangle rect) { return SetComputeScissor(pass.name, rect); }
    bool SetComputePriority(const ComputePassHandle& pass, int priority) { return SetComputePriority(pass.name, priority); }
    bool SetComputePassCondition(const ComputePassHandle& pass, std::function<bool()> condition) { return SetComputePassCondition(pass.name, std::move(condition)); }
    bool SetComputePersistentState(const ComputePassHandle& pass, bool enable) { return SetComputePersistentState(pass.name, enable); }
    bool DestroyComputePass(const ComputePassHandle& pass) { return DestroyComputePass(pass.name); }
    bool DispatchComputePass(const ComputePassHandle& pass) { return DispatchComputePass(pass.name); }
    bool AttachComputePassToGraph(const ComputePassHandle& pass) { return AttachComputePassToGraph(pass.name); }
    bool RemoveComputePassFromGraph(const ComputePassHandle& pass) { return RemoveComputePassFromGraph(pass.name); }
    bool InsertComputePassBefore(const ComputePassHandle& pass, const std::string& beforeNodeName) { return InsertComputePassBefore(pass.name, beforeNodeName); }
    bool InsertComputePassAfter(const ComputePassHandle& pass, const std::string& afterNodeName) { return InsertComputePassAfter(pass.name, afterNodeName); }
    bool AddComputeClass(const ComputePassHandle& pass, const std::string& className) { return AddComputeClass(pass.name, className); }
    bool RemoveComputeClass(const ComputePassHandle& pass, const std::string& className) { return RemoveComputeClass(pass.name, className); }

    bool SetComputeFloat(const ComputePassHandle& pass, const std::string& uniformName, float value) { return SetComputeFloat(pass.name, uniformName, value); }
    bool SetComputeInt(const ComputePassHandle& pass, const std::string& uniformName, int value) { return SetComputeInt(pass.name, uniformName, value); }
    bool SetComputeVector2(const ComputePassHandle& pass, const std::string& uniformName, Vector2 value) { return SetComputeVector2(pass.name, uniformName, value); }
    bool SetComputeVector3(const ComputePassHandle& pass, const std::string& uniformName, Vector3 value) { return SetComputeVector3(pass.name, uniformName, value); }
    bool SetComputeVector4(const ComputePassHandle& pass, const std::string& uniformName, Vector4 value) { return SetComputeVector4(pass.name, uniformName, value); }
    bool SetComputeColor(const ComputePassHandle& pass, const std::string& uniformName, Color value) { return SetComputeColor(pass.name, uniformName, value); }

    bool BindBufferToComputePass(const ComputePassHandle& pass, const BufferHandle& buffer, unsigned int bindingIndex) { return BindBufferToComputePass(pass.name, buffer.name, bindingIndex); }
    bool UnbindBufferFromComputePass(const ComputePassHandle& pass, unsigned int bindingIndex) { return UnbindBufferFromComputePass(pass.name, bindingIndex); }
    bool SwapComputePassBufferBindings(const ComputePassHandle& pass, unsigned int a, unsigned int b) { return SwapComputePassBufferBindings(pass.name, a, b); }
    bool BindUniformBufferToComputePass(const ComputePassHandle& pass, const UniformBufferHandle& buffer, unsigned int bindingIndex) { return BindUniformBufferToComputePass(pass.name, buffer.name, bindingIndex); }
    bool UnbindUniformBufferFromComputePass(const ComputePassHandle& pass, unsigned int bindingIndex) { return UnbindUniformBufferFromComputePass(pass.name, bindingIndex); }

    // New scoped binding overloads for compute pass
    ComputeImageBindingHandle CreateComputeImageBinding(const ComputePassHandle& pass, const std::string& bindingName, const std::string& uniformName, bool readOnly = true, int format = RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, int imageUnit = -1) {
        return CreateComputeImageBinding(pass.name, bindingName, uniformName, readOnly, format, imageUnit);
    }
    bool SetComputeImageTexture(const ComputeImageBindingHandle& binding, Texture2D texture) {
        return SetComputeImageTexture(binding.pass.name, binding.bindingName, texture);
    }
    bool SetComputeImageSource(const ComputeImageBindingHandle& binding, const TextureHandle& source) {
        return SetComputeImageSource(binding.pass.name, binding.bindingName, source.name);
    }
    bool SetComputeImageSource(const ComputeImageBindingHandle& binding, const std::string& sourceName) {
        return SetComputeImageSource(binding.pass.name, binding.bindingName, sourceName);
    }
    bool SetComputeImageRegistrySource(const ComputeImageBindingHandle& binding, const std::string& registryName) {
        return SetComputeImageRegistrySource(binding.pass.name, binding.bindingName, registryName);
    }
    bool SetComputeImageRawTexture(const ComputeImageBindingHandle& binding, unsigned int textureId, unsigned int glInternalFormat) {
        return SetComputeImageRawTexture(binding.pass.name, binding.bindingName, textureId, glInternalFormat);
    }
    bool DestroyComputeImageBinding(const ComputeImageBindingHandle& binding) {
        return DestroyComputeImageBinding(binding.pass.name, binding.bindingName);
    }

    ComputeTextureBindingHandle CreateComputeTextureBinding(const ComputePassHandle& pass, const std::string& bindingName, const std::string& uniformName) {
        return CreateComputeTextureBinding(pass.name, bindingName, uniformName);
    }
    bool SetComputeTexture(const ComputeTextureBindingHandle& binding, Texture2D texture) {
        return SetComputeTexture(binding.pass.name, binding.bindingName, texture);
    }
    bool SetComputeTextureSource(const ComputeTextureBindingHandle& binding, const TextureHandle& source) {
        return SetComputeTextureSource(binding.pass.name, binding.bindingName, source.name);
    }
    bool SetComputeTextureSource(const ComputeTextureBindingHandle& binding, const std::string& sourceName) {
        return SetComputeTextureSource(binding.pass.name, binding.bindingName, sourceName);
    }
    bool SetComputeTextureRegistrySource(const ComputeTextureBindingHandle& binding, const std::string& registryName) {
        return SetComputeTextureRegistrySource(binding.pass.name, binding.bindingName, registryName);
    }
    bool DestroyComputeTextureBinding(const ComputeTextureBindingHandle& binding) {
        return DestroyComputeTextureBinding(binding.pass.name, binding.bindingName);
    }

    // Existing (deprecated) string+pass overloads remain for compatibility.
    bool SetComputeImageTexture(const ComputePassHandle& pass, const std::string& bindingName, Texture2D texture) {
        return SetComputeImageTexture(pass.name, bindingName, texture);
    }
    bool SetComputeImageSource(const ComputePassHandle& pass, const std::string& bindingName, const TextureHandle& source) {
        return SetComputeImageSource(pass.name, bindingName, source.name);
    }
    bool SetComputeImageSource(const ComputePassHandle& pass, const std::string& bindingName, const std::string& sourceName) {
        return SetComputeImageSource(pass.name, bindingName, sourceName);
    }
    bool RemoveComputeImageBinding(const ComputePassHandle& pass, const std::string& bindingName) {
        return DestroyComputeImageBinding(pass.name, bindingName);
    }
    bool SetComputeTexture(const ComputePassHandle& pass, const std::string& bindingName, Texture2D texture) {
        return SetComputeTexture(pass.name, bindingName, texture);
    }
    bool SetComputeTextureSource(const ComputePassHandle& pass, const std::string& bindingName, const TextureHandle& source) {
        return SetComputeTextureSource(pass.name, bindingName, source.name);
    }
    bool SetComputeTextureSource(const ComputePassHandle& pass, const std::string& bindingName, const std::string& sourceName) {
        return SetComputeTextureSource(pass.name, bindingName, sourceName);
    }
    bool DestroyComputeTextureBinding(const ComputePassHandle& pass, const std::string& bindingName) {
        return DestroyComputeTextureBinding(pass.name, bindingName);
    }

    bool CopyComputeImageBindings(const ComputePassHandle& from, const ComputePassHandle& to) {
        return CopyComputeImageBindings(from.name, to.name);
    }
    bool CopyComputeTextureBindings(const ComputePassHandle& from, const ComputePassHandle& to) {
        return CopyComputeTextureBindings(from.name, to.name);
    }
    bool CopyComputeBufferBindings(const ComputePassHandle& from, const ComputePassHandle& to) {
        return CopyComputeBufferBindings(from.name, to.name);
    }
    bool CopyAllComputeBindings(const ComputePassHandle& from, const ComputePassHandle& to) {
        return CopyAllComputeBindings(from.name, to.name);
    }

    bool AttachComputePassToGroup(const ComputeGroupHandle& group, const ComputePassHandle& pass) { return AttachComputePassToGroup(group.name, pass.name); }
    bool RemoveComputePassFromGroup(const ComputeGroupHandle& group, const ComputePassHandle& pass) { return RemoveComputePassFromGroup(group.name, pass.name); }

    // -- Shaders / buffers / framebuffers / groups --
    bool DestroyShader(const ShaderHandle& shader) { return DestroyShader(shader.name); }
    bool ReloadShader(const ShaderHandle& shader) { return ReloadShader(shader.name); }
    bool DestroyComputeShader(const ComputeShaderHandle& shader) { return DestroyComputeShader(shader.name); }
    bool ReloadComputeShader(const ComputeShaderHandle& shader) { return ReloadComputeShader(shader.name); }

    bool DestroyBuffer(const BufferHandle& buffer) { return DestroyBuffer(buffer.name); }
    bool ResizeBuffer(const BufferHandle& buffer, unsigned int newSizeBytes, bool preserveData = true) { return ResizeBuffer(buffer.name, newSizeBytes, preserveData); }
    bool UpdateBuffer(const BufferHandle& buffer, const void* data, unsigned int sizeBytes, unsigned int offset = 0) { return UpdateBuffer(buffer.name, data, sizeBytes, offset); }
    bool ReadBuffer(const BufferHandle& buffer, void* outData, unsigned int sizeBytes, unsigned int offset = 0) { return ReadBuffer(buffer.name, outData, sizeBytes, offset); }

    bool DestroyUniformBuffer(const UniformBufferHandle& buffer) { return DestroyUniformBuffer(buffer.name); }
    bool UpdateUniformBuffer(const UniformBufferHandle& buffer, const void* data, unsigned int sizeBytes, unsigned int offset = 0) { return UpdateUniformBuffer(buffer.name, data, sizeBytes, offset); }

    bool DestroyFramebuffer(const FramebufferHandle& fb) { return DestroyFramebuffer(fb.name); }
    bool ResizeFramebuffer(const FramebufferHandle& fb, int width, int height) { return ResizeFramebuffer(fb.name, width, height); }
    void BindFramebuffer(const FramebufferHandle& fb) { BindFramebuffer(fb.name); }
    bool BlitFramebuffer(const FramebufferHandle& src, Rectangle srcRect, const FramebufferHandle& dst, Rectangle dstRect, int mask) {
        return BlitFramebuffer(src.name, srcRect, dst.name, dstRect, mask);
    }

    bool DestroyProcessGroup(const GroupHandle& group) { return DestroyProcessGroup(group.name); }
    bool SetGroupEnabled(const GroupHandle& group, bool enabled) { return SetGroupEnabled(group.name, enabled); }
    bool SetGroupSource(const GroupHandle& group, const std::string& sourceName) { return SetGroupSource(group.name, sourceName); }
    bool SetGroupCondition(const GroupHandle& group, std::function<bool()> condition) { return SetGroupCondition(group.name, std::move(condition)); }

    bool DestroyComputeGroup(const ComputeGroupHandle& group) { return DestroyComputeGroup(group.name); }
    bool SetComputeGroupEnabled(const ComputeGroupHandle& group, bool enabled) { return SetComputeGroupEnabled(group.name, enabled); }
    bool SetComputeGroupCondition(const ComputeGroupHandle& group, std::function<bool()> condition) { return SetComputeGroupCondition(group.name, std::move(condition)); }
    bool SetComputeGroupPriority(const ComputeGroupHandle& group, int priority) { return SetComputeGroupPriority(group.name, priority); }
    bool AttachComputeGroupToGraph(const ComputeGroupHandle& group) { return AttachComputeGroupToGraph(group.name); }
    bool RemoveComputeGroupFromGraph(const ComputeGroupHandle& group) { return RemoveComputeGroupFromGraph(group.name); }
    bool InsertComputeGroupBefore(const ComputeGroupHandle& group, const std::string& beforeNodeName) { return InsertComputeGroupBefore(group.name, beforeNodeName); }
    bool InsertComputeGroupAfter(const ComputeGroupHandle& group, const std::string& afterNodeName) { return InsertComputeGroupAfter(group.name, afterNodeName); }

    // -- Dependencies --
    bool AddDependency(const PassHandle& source, const PassHandle& target) { return AddDependency(source.name, target.name); }
    bool RemoveDependency(const PassHandle& source, const PassHandle& target) { return RemoveDependency(source.name, target.name); }
    bool AddDependency(const GeometryPassHandle& source, const GeometryPassHandle& target) { return AddDependency(source.name, target.name); }
    bool RemoveDependency(const GeometryPassHandle& source, const GeometryPassHandle& target) { return RemoveDependency(source.name, target.name); }
    bool AddDependency(const ComputePassHandle& source, const ComputePassHandle& target) { return AddDependency(source.name, target.name); }
    bool RemoveDependency(const ComputePassHandle& source, const ComputePassHandle& target) { return RemoveDependency(source.name, target.name); }
    bool AddDependency(const GroupHandle& source, const GroupHandle& target) { return AddDependency(source.name, target.name); }
    bool RemoveDependency(const GroupHandle& source, const GroupHandle& target) { return RemoveDependency(source.name, target.name); }
    bool AddDependency(const ComputeGroupHandle& source, const ComputeGroupHandle& target) { return AddDependency(source.name, target.name); }
    bool RemoveDependency(const ComputeGroupHandle& source, const ComputeGroupHandle& target) { return RemoveDependency(source.name, target.name); }

    // ------------------------------------------------------------------
    // Handle-typed Find* overloads
    // ------------------------------------------------------------------
    ProcessPass* FindPass(const PassHandle& handle) { return FindPass(handle.name); }
    const ProcessPass* FindPass(const PassHandle& handle) const { return FindPass(handle.name); }

    ProcessGroup* FindGroup(const GroupHandle& handle) { return FindGroup(handle.name); }
    const ProcessGroup* FindGroup(const GroupHandle& handle) const { return FindGroup(handle.name); }

    ComputeGroup* FindComputeGroup(const ComputeGroupHandle& handle) { return FindComputeGroup(handle.name); }
    const ComputeGroup* FindComputeGroup(const ComputeGroupHandle& handle) const { return FindComputeGroup(handle.name); }

    Framebuffer* FindFramebuffer(const FramebufferHandle& handle) { return FindFramebuffer(handle.name); }
    const Framebuffer* FindFramebuffer(const FramebufferHandle& handle) const { return FindFramebuffer(handle.name); }

    ComputeProgram* FindComputeProgram(const ComputeShaderHandle& handle) { return FindComputeProgram(handle.name); }
    const ComputeProgram* FindComputeProgram(const ComputeShaderHandle& handle) const { return FindComputeProgram(handle.name); }

    ShaderBuffer* FindBuffer(const BufferHandle& handle) { return FindBuffer(handle.name); }
    const ShaderBuffer* FindBuffer(const BufferHandle& handle) const { return FindBuffer(handle.name); }

    UniformBuffer* FindUniformBuffer(const UniformBufferHandle& handle) { return FindUniformBuffer(handle.name); }
    const UniformBuffer* FindUniformBuffer(const UniformBufferHandle& handle) const { return FindUniformBuffer(handle.name); }

    ComputePass* FindComputePass(const ComputePassHandle& handle) { return FindComputePass(handle.name); }
    const ComputePass* FindComputePass(const ComputePassHandle& handle) const { return FindComputePass(handle.name); }

    GeometryPass* FindGeometryPass(const GeometryPassHandle& handle) { return FindGeometryPass(handle.name); }
    const GeometryPass* FindGeometryPass(const GeometryPassHandle& handle) const { return FindGeometryPass(handle.name); }

    ProcessShader* FindShader(const ShaderHandle& handle) { return FindShader(handle.name); }
    const ProcessShader* FindShader(const ShaderHandle& handle) const { return FindShader(handle.name); }

    // Additional handle-typed getters for output textures
    const Texture2D* GetPassOutputTexture(const PassHandle& handle) const { return GetPassOutputTexture(handle.name); }
    const Texture2D* GetGroupOutputTexture(const GroupHandle& handle) const { return GetGroupOutputTexture(handle.name); }

private:
    enum class RenderNodeType {
        Pass,
        Group,
        ComputeGroup,
        Compute,
        Geometry
    };

    struct RenderNode {
        RenderNodeType type;
        std::string name;
    };

    void ApplyPassRenderState(const PassRenderState& state);
    void RestorePassRenderState(const PassRenderState& state);

    // Shared tail of CreateComputeShader/CreateComputeShaderFromSource --
    // see definition for details.
    ComputeShaderHandle FinishComputeShaderCreation(const std::string& name,
                                                     const std::string& sourcePath,
                                                     unsigned int csId);

    void ReleasePassGpuResources(ProcessPass& pass);
    void EnsureAllPassOutputsAllocated();
    void EnsureBarrierForTexture(unsigned int textureId);
    Texture2D ResolveFramebufferColorTexture(const std::string& framebufferName, std::size_t attachmentIndex = 0) const;
    static void EmitMemoryBarrier(unsigned int barrierBits);
    void ReapplyUniforms(ProcessPass& pass);
    bool AllocatePassOutput(ProcessPass& pass);
    bool AllocateGroupOutput(ProcessGroup& group);
    void ReapplyComputeUniforms(ComputePass& pass);
    void SwapHistory(ProcessPass& pass, Texture2D sourceTexture);
    int GetNodePriority(const std::string& name) const;
    void SetError(const std::string& message) const;
    static std::string NodeTypeToString(RenderNodeType type);

    std::vector<std::pair<std::string, std::string>> dependencies_;
    bool automaticOrdering_ = false;
    void ComputeExecutionOrder(std::vector<RenderNode>& outOrder) const;

    ProcessGroup* FindGroupContainingPass(const std::string& passName);
    const ProcessGroup* FindGroupContainingPass(const std::string& passName) const;
    bool IsPassGrouped(const std::string& passName) const;

    RenderTexture2D AcquirePooledTexture(int width, int height, TextureFilter filter, int format, bool withDepth);
    void ReleaseToPool(RenderTexture2D texture, int format, bool withDepth);

    int width_ = 0;
    int height_ = 0;
    RenderTexture2D sceneTexture_{};
    RenderTexture2D pingPongBuffers_[2]{};
    Texture2D currentOutput_{};

    std::vector<ProcessShader> shaders_;
    std::vector<ProcessPass> passes_;
    std::vector<ProcessGroup> groups_;
    std::vector<ProcessGroupTemplate> groupTemplates_;
    std::vector<ProcessGroupInstance> groupInstances_;
    std::vector<RenderNode> renderOrder_;

    std::vector<ComputeProgram> computePrograms_;
    std::vector<ComputePass> computePasses_;
    std::vector<ComputeGroup> computeGroups_;
    std::vector<ShaderBuffer> buffers_;
    std::vector<Framebuffer> framebuffers_;
    std::vector<GeometryPass> geometryPasses_;
    std::vector<UniformBuffer> uniformBuffers_;
    std::vector<PooledRenderTexture> renderTexturePool_;

    std::unordered_map<std::string, std::size_t> shadersIndex_;
    std::unordered_map<std::string, std::size_t> passesIndex_;
    std::unordered_map<std::string, std::size_t> computeProgramsIndex_;
    std::unordered_map<std::string, std::size_t> computePassesIndex_;
    std::unordered_map<std::string, std::size_t> buffersIndex_;
    std::unordered_map<std::string, std::size_t> framebuffersIndex_;
    std::unordered_map<std::string, std::size_t> geometryPassesIndex_;

    std::unordered_set<unsigned int> computeWrittenTextures_;
    std::unordered_set<unsigned int> barrierFlushedTextures_; 

    bool previewAllPasses_ = false;
    bool profilingEnabled_ = false;
    std::unordered_map<std::string, unsigned int> queryIds_;
    std::unordered_map<std::string, float> gpuTimes_;
    std::unordered_map<std::string, RenderTexture2D> sceneTextures_;

    mutable std::string lastError_;
    unsigned int currentShaderId_ = 0;

    // Shared VAO for every attributeless (SSBO/gl_VertexID-driven)
    // GeometryPass -- created once in Initialize(), bound around whichever
    // pass currently has attributelessDraw set, never touched otherwise.
    // One VAO is enough for every such pass since it has no vertex
    // attributes of its own to configure; it exists purely because
    // desktop GL core profile refuses to draw with no VAO bound at all.
    unsigned int attributelessVAO_ = 0;

    // ------------------------------------------------------------------
    // Kernel shaders / automatic uniforms (see public section above)
    // ------------------------------------------------------------------
    ShaderKernelLibrary kernelLibrary_;
    // Split so the same name can mean a different texture depending on
    // whether a kernel binds it as `image2D` or `sampler2D` -- see
    // RegisterImage/RegisterSampler above.
    std::unordered_map<std::string, Texture2D> kernelImageRegistry_;
    std::unordered_map<std::string, Texture2D> kernelSamplerRegistry_;
    std::unordered_map<std::string, BufferHandle> kernelBufferRegistry_;
    // compute pass name -> kernel name, so Apply() knows which KernelInfo
    // to reflect against when auto-pushing global uniforms to that pass.
    std::unordered_map<std::string, std::string> computePassKernelName_;

    // Like BindBufferToComputePass, but doesn't require a buffer named
    // `blockName` to exist yet -- used by CreateComputePassFromKernel so
    // a buffer bound by its kernel block name can be registered (via
    // RegisterBuffer) before or after the pass is created. Resolved
    // through kernelBufferRegistry_ at dispatch time; see the SSBO bind
    // loop in DispatchComputePass.
    bool BindBufferSlotByRegistryName(const std::string& passName, const std::string& blockName, unsigned int bindingIndex);

    struct GlobalUniformEntry {
        UniformValue value;
        uint64_t version = 0;
    };
    uint64_t nextGlobalUniformVersion_ = 0;
    std::unordered_map<std::string, GlobalUniformEntry> globalUniforms_;
    // pass name -> {uniform name -> last version pushed to that pass}.
    std::unordered_map<std::string, std::unordered_map<std::string, uint64_t>> globalUniformLastAppliedToPass_;

    template <typename T>
    void StoreGlobalUniform(const std::string& name, T value) {
        UniformValue newValue{value};
        auto it = globalUniforms_.find(name);
        if (it == globalUniforms_.end()) {
            globalUniforms_.emplace(name, GlobalUniformEntry{newValue, ++nextGlobalUniformVersion_});
            return;
        }
        if (UniformValueEquals(it->second.value, newValue)) return; // unchanged -- version stays put
        it->second.value = newValue;
        it->second.version = ++nextGlobalUniformVersion_;
    }
    void ApplyGlobalUniformsToKernelPasses();
};