#include "raylib.h"
int main(){
    InitWindow(1200,800,"Raytraced audio");
    SetTargetFPS(60);
    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(BLACK);
        EndDrawing();
    }
    CloseWindow();
}