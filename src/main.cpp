#include "raylib.h"
#include "raymath.h"
#include "render_graph.h"
#include <vector>

struct GameObject {
    Model      model;
    Vector3    position = Vector3Zero();
    Quaternion rotation = QuaternionIdentity();
    Vector3    scale    = Vector3One();
    GameObject(Model m) : model(m) { UpdateTransform(); }

    GameObject(Model m, Vector3 pos, Quaternion rot, Vector3 scl)
        : model(m), position(pos), rotation(rot), scale(scl)
    {
        UpdateTransform();
    }

    // Rebuild the world matrix from position / rotation / scale.
    // Order: Scale * Rotation * Translation  (raylib uses row-vector convention)
    void UpdateTransform()
    {
        Matrix matScale = MatrixScale(scale.x, scale.y, scale.z);
        Matrix matRot   = QuaternionToMatrix(rotation);
        Matrix matTrans = MatrixTranslate(position.x, position.y, position.z);
        model.transform = MatrixMultiply(MatrixMultiply(matScale, matRot), matTrans);
    }

    // Draw the model using the composed transform.
    // DrawModel already multiplies model.transform by position/scale,
    // so we pass zero position and unit scale and only use the matrix.
    void Draw(Color tint = WHITE) const
    {
        DrawModel(model, Vector3Zero(), 1.0f, tint);
    }
};

struct AudioRay {
    Ray          ray;
    RayCollision collision;
    int          reflection_status = 0;
};

Vector3 Reflect(Vector3 incoming, Vector3 normal)
{
    return Vector3Subtract(incoming,
                           Vector3Scale(normal, 2.0f * Vector3DotProduct(incoming, normal)));
}

// Returns true on hit and fills `out_collision`. Does not touch anything else.
static bool RayHitsModel(const Ray& ray, const Model& model,
                         RayCollision& out_collision)
{
    bool hit_any = false;
    float closest = 1e6;
    for (int i = 0; i < model.meshCount; i++) {
        RayCollision c = GetRayCollisionMesh(ray, model.meshes[i], model.transform);
        if (c.hit && c.distance < closest) {
            closest       = c.distance;
            out_collision = c;
            hit_any       = true;
        }
    }
    if (!hit_any) out_collision.hit = false;
    return hit_any;
}

int main()
{
    InitWindow(1200, 800, "Raytraced audio");

    Model room    = LoadModel("models/Room.glb");
    Model man     = LoadModel("models/man.glb");
    Model speaker = LoadModel("models/Speaker.glb");
    Camera cam = { 0 };
    cam.position   = { 1.0f, 3.0f, 1.0f };
    cam.target     = { 0.0f, 0.0f, 0.0f };
    cam.up         = { 0.0f, 1.0f, 0.0f };
    cam.fovy       = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    const int   num_rays    = 360;
    const float model_scale = 0.1f;
    GameObject listen_src(man);
    listen_src.scale=Vector3(0.1f,0.1f,0.1f);
    listen_src.UpdateTransform();
    GameObject room_obj (room);
    room_obj.scale=listen_src.scale;
    room_obj.UpdateTransform();
    GameObject audio_src(speaker);
    audio_src.scale=listen_src.scale;
    audio_src.UpdateTransform();

    const int   MAX_BOUNCES = 5;
    const float NUDGE       = 1e-3f;
    Vector3 speaker_pos=Vector3{0.3,0.0,-0.7};
    Vector3 man_pos=Vector3Zero();
    std::vector<AudioRay> rays;
    rays.reserve(num_rays * (MAX_BOUNCES + 1));

    for (int i = 0; i < num_rays; i++) {
        float theta = (360.0f / (float)num_rays) * (float)i;
        AudioRay ar;
        ar.ray.position  = Vector3{ 0.0f, 0.1f, 0.0f };
        ar.ray.direction = Vector3{ cosf(theta * DEG2RAD), 0.0f, sinf(theta * DEG2RAD) };
        rays.push_back(ar);
    }

    for (size_t i = 0; i < rays.size(); i++) {
        AudioRay& ar = rays[i];
        if (ar.reflection_status >= MAX_BOUNCES) continue;
        RayCollision hit;
        if (RayHitsModel(ar.ray, room_obj.model, hit)) {
            ar.collision = hit;
            AudioRay next;
            next.ray.position  = Vector3Add(hit.point,Vector3Scale(hit.normal, NUDGE));
            next.ray.direction = Reflect(Vector3Normalize(ar.ray.direction), hit.normal);
            next.reflection_status = ar.reflection_status + 1;
            rays.push_back(next);
        }
    }

    // ---- Render ----
    RenderGraph rg;
    rg.Initialize(1200, 800);
    rg.CreateDefaultShader("default");
    rg.CreateGeometryPass("static_pass", "default");

    rg.SetGeometryPassCallback("static_pass", [&]() {
        BeginMode3D(cam);
            
            room_obj.Draw();
            listen_src.Draw();
            audio_src.Draw();

            for (auto& ar : rays) {
                // Color by bounce count, or by whether it hit at all:
                Color c =GREEN;
                if(ar.collision.hit){
                    DrawLine3D(ar.ray.position,ar.collision.point, c);
                }
            }
        EndMode3D();
    });

    rg.AttachGeometryPassToGraph("static_pass");

    SetTargetFPS(60);
    while (!WindowShouldClose())
    {
        BeginDrawing();
            ClearBackground(BLACK);
            rg.Apply();
        EndDrawing();
    }

    UnloadModel(room);
    UnloadModel(man);
    CloseWindow();
}