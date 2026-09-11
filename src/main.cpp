#include "raylib.h"
#include "raymath.h"
#include "render_graph.h"
#include <vector>

struct GameObject {
    Model   model;
    Matrix  transform;
    GameObject(Model m, Matrix t) : model(m), transform(t) {}
    GameObject(Model m)           : model(m), transform(MatrixIdentity()) {}
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
static bool RayHitsModel(const Ray& ray, const Model& model, Matrix transform,
                         RayCollision& out_collision)
{
    bool hit_any = false;
    float closest = 1e6;
    for (int i = 0; i < model.meshCount; i++) {
        RayCollision c = GetRayCollisionMesh(ray, model.meshes[i], transform);
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

    Model room = LoadModel("models/Room.glb");
    Model man  = LoadModel("models/man.glb");

    Camera cam = { 0 };
    cam.position   = { 1.0f, 3.0f, 1.0f };
    cam.target     = { 0.0f, 0.0f, 0.0f };
    cam.up         = { 0.0f, 1.0f, 0.0f };
    cam.fovy       = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    const int   num_rays    = 120;
    const float model_scale = 0.1f;
    const Matrix model_transform = MatrixScale(model_scale, model_scale, model_scale);

    GameObject audio_src(man,  model_transform);
    GameObject room_obj (room, model_transform);

    // ---- Compute reflections ONCE, outside the render loop ----
    const int   MAX_BOUNCES = 5;
    const float NUDGE       = 1e-3f;

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
        if (RayHitsModel(ar.ray, room, model_transform, hit)) {
            ar.collision = hit;

            AudioRay next;
            next.ray.position  = Vector3Add(hit.point,
                                            Vector3Scale(hit.normal, NUDGE));
            next.ray.direction = Reflect(Vector3Normalize(ar.ray.direction), hit.normal);
            next.reflection_status = ar.reflection_status + 1;
            rays.push_back(next);   // safe here: we're using an index, not an iterator
        }
    }

    // ---- Render ----
    RenderGraph rg;
    rg.Initialize(1200, 800);
    rg.CreateDefaultShader("default");
    rg.CreateGeometryPass("static_pass", "default");

    rg.SetGeometryPassCallback("static_pass", [&]() {
        BeginMode3D(cam);
            DrawModel(room_obj.model,  Vector3Zero(), model_scale, WHITE);
            DrawModel(audio_src.model, Vector3Zero(), model_scale, WHITE);

            for (auto& ar : rays) {
                // Color by bounce count, or by whether it hit at all:
                Color c = (ar.reflection_status == 0) ? GREEN
                        : (ar.reflection_status == 1) ? RED
                        : (ar.reflection_status == 2) ? ORANGE
                        : (ar.reflection_status == 3) ? YELLOW
                        : BLUE;
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