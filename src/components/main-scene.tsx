"use client"

import { useRef, useEffect, useCallback, useState } from "react"
import Image from "next/image"
import Link from "next/link"
import { Button } from "@/components/ui/button"

import { Engine, EngineStats, Quat, Vec3 } from "reze-engine"

import { MotionCapture } from "./motion-capture"
import { BoneState } from "@/lib/solver"

export default function MainScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const [modelLoaded, setModelLoaded] = useState(false)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [stats, setStats] = useState<EngineStats | null>(null)

  const initEngine = useCallback(async () => {
    if (canvasRef.current) {
      // Initialize engine
      try {
        const engine = new Engine(canvasRef.current, {
          ambientColor: new Vec3(0.85, 0.9, 1.0),
        })
        engineRef.current = engine
        await engine.init()
        await engine.loadModel("/models/塞尔凯特/塞尔凯特.pmx")
        setModelLoaded(true)
        engine.runRenderLoop(() => {
          setStats(engine.getStats())
        })
      } catch (error) {
        setEngineError(error instanceof Error ? error.message : "Unknown error")
      }
    }
  }, [])

  useEffect(() => {
    void (async () => {
      initEngine()
    })()

    // Cleanup on unmount
    return () => {
      if (engineRef.current) {
        engineRef.current.dispose()
      }
    }
  }, [initEngine])

  const applyPose = useCallback(
    (boneStates: BoneState[]) => {
      if (!engineRef.current) return
      engineRef.current.rotateBones(
        boneStates.map((bone) => bone.name),
        boneStates.map((bone) => new Quat(bone.rotation.x, bone.rotation.y, bone.rotation.z, bone.rotation.w)),
        60
      )
    },
    [engineRef]
  )

  return (
    <div className="w-full h-full">
      <div className="absolute p-4 top-0 left-0 w-full z-10 flex flex-row items-center justify-end gap-4">
        {stats && <div className="text-white z-10 font-medium text-sm">FPS: {stats.fps}</div>}

        <Button size="icon" asChild className="bg-white text-black size-7 rounded-full z-10 hover:bg-gray-200">
          <Link href="https://github.com/AmyangXYZ/MiKaPo" target="_blank">
            <Image src="/github-mark.svg" alt="GitHub" width={18} height={18} />
          </Link>
        </Button>
      </div>

      <div className="absolute p-6 bottom-0 left-0 w-full z-10 flex flex-row items-center justify-start gap-4">
        <div className="text-white z-10 font-medium text-sm">
          Powered by{" "}
          <Link
            href="https://reze.one"
            target="_blank"
            className="text-blue-400 shadow-lg bg-white/10 rounded-md px-2 py-1"
          >
            Reze Engine
          </Link>
        </div>
      </div>

      <MotionCapture applyPose={applyPose} modelLoaded={modelLoaded} />

      {engineError && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 font-medium text-white z-10">
          {engineError}
        </div>
      )}
      <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full z-1 outline-none" />
    </div>
  )
}
