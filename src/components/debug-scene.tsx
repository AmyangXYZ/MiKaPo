import { useCallback, useEffect, useRef } from "react"
import { ArcRotateCamera, Color3, Color4, Engine, LinesMesh, MeshBuilder, Scene, Vector3 } from "@babylonjs/core"
import { GridMaterial } from "@babylonjs/materials"
import { HolisticLandmarker, HolisticLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision"

function DebugScene({ landmarks }: { landmarks: HolisticLandmarkerResult | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<Scene | null>(null)
  const engineRef = useRef<Engine | null>(null)
  const lineMeshesRef = useRef<Map<string, LinesMesh>>(new Map())

  useEffect(() => {
    const createScene = async () => {
      if (!canvasRef.current) return
      const engine = new Engine(canvasRef.current, true, {}, true)
      engineRef.current = engine
      const scene = new Scene(engine)
      scene.clearColor = new Color4(0, 0, 0, 1)
      sceneRef.current = scene

      const camera = new ArcRotateCamera("ArcRotateCamera", 0, 0, 45, new Vector3(0, 0, 0), scene)
      camera.setPosition(new Vector3(0, 1.5, -2))
      camera.attachControl(canvasRef.current, false)
      camera.inertia = 0.8
      camera.speed = 10

      const groundMaterial = new GridMaterial("groundMaterial", scene)
      groundMaterial.majorUnitFrequency = 5
      groundMaterial.minorUnitVisibility = 0.5
      groundMaterial.gridRatio = 1
      groundMaterial.opacity = 0.99
      groundMaterial.useMaxLine = true

      const ground = MeshBuilder.CreateGround("ground", { width: 4, height: 4, updatable: false }, scene)
      ground.material = groundMaterial

      engine.runRenderLoop(() => {
        engine.resize()
        scene!.render()
      })
    }

    createScene()
    return () => {
      engineRef.current?.dispose()
    }
  }, [])

  const drawConnections = useCallback(
    (
      landmarks: NormalizedLandmark[] | null,
      connections: { start: number; end: number }[],
      color: Color3,
      meshNamePrefix: string
    ): void => {
      if (!sceneRef.current) return

      if (!landmarks || landmarks.length === 0) return

      // Only clear existing meshes when we have new landmarks to draw
      lineMeshesRef.current.forEach((mesh, key) => {
        if (key.startsWith(meshNamePrefix)) {
          mesh.dispose()
          lineMeshesRef.current.delete(key)
        }
      })

      const points = landmarks.map((lm) => new Vector3(lm.x, -lm.y, lm.z))
      connections.forEach((connection, index) => {
        const start = points[connection.start]
        const end = points[connection.end]
        const meshKey = `${meshNamePrefix}_${index}`

        const lineMesh = MeshBuilder.CreateLines(
          `debug_lines_${meshNamePrefix}_${index}`,
          { points: [start, end] },
          sceneRef.current
        )
        lineMesh.color = color
        lineMeshesRef.current.set(meshKey, lineMesh)
      })
    },
    []
  )

  useEffect(() => {
    if (!sceneRef.current || !landmarks) return

    drawConnections(landmarks.poseWorldLandmarks[0], HolisticLandmarker.POSE_CONNECTIONS, new Color3(1, 1, 1), "pose")
    drawConnections(
      landmarks.leftHandWorldLandmarks[0],
      HolisticLandmarker.HAND_CONNECTIONS,
      new Color3(1, 1, 1),
      "left_hand"
    )
    drawConnections(
      landmarks.rightHandWorldLandmarks[0],
      HolisticLandmarker.HAND_CONNECTIONS,
      new Color3(1, 1, 1),
      "right_hand"
    )
  }, [landmarks, drawConnections])

  return <canvas ref={canvasRef} className="pt-4 max-w-sm w-full h-[400px] outline-none"></canvas>
}

export default DebugScene
