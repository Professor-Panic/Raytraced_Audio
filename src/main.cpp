#include "raylib.h"
#include "raymath.h"
#include "render_graph.h"
int main(){
    InitWindow(1200,800,"Raytraced audio");

    Model room = LoadModel("models/Room.glb");

    Camera cam = { 0 };
    cam.position   = { 1.0f, 3.0f, 1.0f };
    cam.target     = { 0.0f, 0.0f, 0.0f };
    cam.up         = { 0.0f, 1.0f, 0.0f };
    cam.fovy       = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;
    RenderGraph rg;
    rg.Initialize(1200,800);
    rg.CreateDefaultShader("default");
    rg.CreateGeometryPass("static_pass","default");
    rg.SetGeometryPassCallback("static_pass",[&](){
        BeginMode3D(cam);
            DrawModel(room, Vector3Zero(), 0.1f, WHITE);
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
    CloseWindow();
}