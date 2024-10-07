import { useEffect, useRef } from "react"
import { ArcRotateCamera, Color3, Color4, Engine, MeshBuilder, Scene, Vector3 } from "@babylonjs/core"
import { GridMaterial } from "@babylonjs/materials"
import { HolisticLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision"

import "@babylonjs/core/Engines/shaderStore"

function DebugScene({
  pose,
  leftHand,
  rightHand,
}: {
  pose: NormalizedLandmark[] | null
  leftHand: NormalizedLandmark[] | null
  rightHand: NormalizedLandmark[] | null
}): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<Scene | null>(null)
  const engineRef = useRef<Engine | null>(null)
  const cameraRef = useRef<ArcRotateCamera | null>(null)

  useEffect(() => {
    const createScene = async (canvas: HTMLCanvasElement): Promise<Scene> => {
      const engine = new Engine(canvas, true, {}, true)
      engineRef.current = engine
      const scene = new Scene(engine)
      scene.clearColor = new Color4(0, 0, 0, 0)

      const camera = new ArcRotateCamera("ArcRotateCamera", 0, 0, 45, new Vector3(0, 12, 0), scene)
      camera.setPosition(new Vector3(0, 15, -27))
      camera.attachControl(canvas, false)
      camera.inertia = 0.8
      camera.speed = 10
      cameraRef.current = camera

      const groundMaterial = new GridMaterial("groundMaterial", scene)
      groundMaterial.majorUnitFrequency = 5
      groundMaterial.minorUnitVisibility = 0.5
      groundMaterial.gridRatio = 2
      groundMaterial.opacity = 0.99
      groundMaterial.useMaxLine = true

      const ground = MeshBuilder.CreateGround("ground", { width: 40, height: 40, updatable: false }, scene)
      ground.material = groundMaterial

      engine.runRenderLoop(() => {
        engine.resize()
        scene!.render()
      })
      return scene
    }

    if (canvasRef.current) {
      createScene(canvasRef.current).then((scene) => {
        sceneRef.current = scene
      })
    }
    return () => {
      engineRef.current?.dispose()
    }
  }, [])

  useEffect(() => {
    const drawDebug = (): void => {
      const scene = sceneRef.current
      const scale = 12
      const yOffset = 10

      const drawConnections = (
        landmarks: NormalizedLandmark[] | null,
        connections: { start: number; end: number }[],
        color: Color3,
        meshNamePrefix: string
      ): void => {
        if (!landmarks || landmarks.length === 0) return
        const points = landmarks.map((lm) => new Vector3(lm.x * scale, -lm.y * scale + yOffset, lm.z * scale))
        connections.forEach((connection, index) => {
          // skip hand connections from pose landmarks
          if (meshNamePrefix === "pose") {
            if ((connection.start >= 17 && connection.start <= 22) || (connection.end >= 17 && connection.end <= 22))
              return
          }

          const start = points[connection.start]
          const end = points[connection.end]
          const lineMesh = MeshBuilder.CreateLines(
            `debug_lines_${meshNamePrefix}_${index}`,
            { points: [start, end] },
            scene
          )
          lineMesh.color = color
        })
      }

      // Remove previous debug lines
      scene!.meshes.filter((mesh) => mesh.name.startsWith("debug_lines")).forEach((mesh) => mesh.dispose())

      // Draw new debug lines
      drawConnections(pose, HolisticLandmarker.POSE_CONNECTIONS, new Color3(1, 1, 1), "pose")
      drawConnections(leftHand, HolisticLandmarker.HAND_CONNECTIONS, new Color3(1, 1, 1), "left_hand")
      drawConnections(rightHand, HolisticLandmarker.HAND_CONNECTIONS, new Color3(1, 1, 1), "right_hand")
    }

    if (sceneRef.current) {
      drawDebug()
    }
  }, [pose, leftHand, rightHand])
  return (
    <>
      <canvas ref={canvasRef} className="debug-scene"></canvas>
    </>
  )
}

export default DebugScene
