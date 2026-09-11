#include "raylib.h"
#include "raymath.h"
#include "render_graph.h"
#include <vector>

// Helper: returns true if the ray hits any mesh of the model
struct GameObject{
    Model model;
    Matrix transform;
    GameObject(Model model,Matrix transform):model(model),transform(transform){}
    GameObject(Model model):model(model),transform(MatrixIdentity()){}
};
struct AudioRay {
    Ray  ray;
    RayCollision collision;
};
static bool RayHitsModel(AudioRay& ray, const Model& model, Matrix transform)
{
    for (int i = 0; i < model.meshCount; i++) {
        ray.collision= GetRayCollisionMesh(ray.ray, model.meshes[i], transform);
        if (ray.collision.hit) return true;
    }
    return false;
}
int main(){
    InitWindow(1200,800,"Raytraced audio");

    Model room = LoadModel("models/Room.glb");
    Model man  = LoadModel("models/man.glb");
    Camera cam = { 0 };
    cam.position   = { 1.0f, 3.0f, 1.0f };
    cam.target     = { 0.0f, 0.0f, 0.0f };
    cam.up         = { 0.0f, 1.0f, 0.0f };
    cam.fovy       = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    const int num_rays = 12;
    const float model_scale = 0.1f;
    const Matrix model_transform = MatrixScale(model_scale, model_scale, model_scale);
    GameObject audio_src=GameObject(man,model_transform);
    GameObject room_obj=GameObject(room,model_transform);
    // Ray + its current hit state (color switch)
    std::vector<AudioRay> rays;
    rays.reserve(num_rays);
    Vector3 man_pos=Vector3Zero();
    for (int i = 0; i < num_rays; i++) {
        float theta = (360.0f / (float)num_rays) * (float)i;
        AudioRay ar;
        ar.ray.position  = man_pos;
        ar.ray.direction = Vector3{ cos(theta * DEG2RAD), 0.0f, sin(theta * DEG2RAD) };
        rays.push_back(ar);
    }
    RenderGraph rg;
    rg.Initialize(1200,800);
    rg.CreateDefaultShader("default");
    rg.CreateGeometryPass("static_pass","default");
    rg.SetGeometryPassCallback("static_pass",[&](){
        BeginMode3D(cam);
            DrawModel(room_obj.model, Vector3Zero(), model_scale, WHITE);
            DrawModel(audio_src.model,  man_pos, model_scale, WHITE);
            for (auto& ar : rays) {
                // --- color switch: red if it hits something, green otherwise ---
                bool hit_man  = RayHitsModel(ar, man,  model_transform);
                bool hit_room = RayHitsModel(ar, room, model_transform);
                ar.collision.hit = hit_man || hit_room;
                DrawRay(ar.ray, ar.collision.hit ? RED : GREEN);
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