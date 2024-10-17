import { Pause, PlayArrow, FileUpload } from "@mui/icons-material"
import { Button, FormControl, FormControlLabel, IconButton, Radio, RadioGroup, Slider, Typography } from "@mui/material"
import { useMemo, useRef } from "react"

const availableAnimations = ["Stand", "Zyy", "Miku", "iKun1", "Man", "0-540"]

function Animation({
  isPlaying,
  setIsPlaying,
  animationDuration,
  setSelectedAnimation,
  currentAnimationTime,
  setAnimationSeekTime,
}: {
  isPlaying: boolean
  setIsPlaying: (isPlaying: boolean) => void
  animationDuration: number
  setSelectedAnimation: (animation: string) => void
  currentAnimationTime: number
  setAnimationSeekTime: (time: number) => void
}): JSX.Element {
  const formatTime = (time: number) => {
    time = Math.round(time)
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes.toString().padStart(1, "0")}:${seconds.toString().padStart(2, "0")}`
  }
  const marks = useMemo(() => {
    return [
      { value: 0, label: formatTime(currentAnimationTime) },
      {
        value: animationDuration,
        label: "-" + formatTime(Math.round(animationDuration) - Math.round(currentAnimationTime)),
      },
    ]
  }, [currentAnimationTime, animationDuration])

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleVMDUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const url = URL.createObjectURL(file)
      setSelectedAnimation(url)
    }
  }

  return (
    <FormControl className="animation">
      <Button
        variant="contained"
        startIcon={<FileUpload />}
        onClick={() => fileInputRef.current?.click()}
        sx={{
          margin: "auto",
          backgroundColor: "#a2c9f5",
          color: "black",
          "&:hover": { backgroundColor: "#7ab1f1" },
          marginBottom: ".3rem",
        }}
      >
        Upload VMD
      </Button>
      <input type="file" ref={fileInputRef} onChange={handleVMDUpload} accept=".vmd" style={{ display: "none" }} />

      <RadioGroup
        aria-labelledby="demo-radio-buttons-group-label"
        defaultValue=""
        name="radio-buttons-group"
        onChange={(_, value) => setSelectedAnimation(`/animation/${value}.vmd`)}
        sx={{ display: "flex", margin: "auto" }}
      >
        {availableAnimations.map((animation) => (
          <FormControlLabel
            key={animation}
            value={animation}
            control={<Radio sx={{ color: "#a2c9f5", "&.Mui-checked": { color: "#a2c9f5" } }} size="small" />}
            label={<Typography sx={{ fontSize: ".9rem" }}>{animation}</Typography>}
          />
        ))}
      </RadioGroup>

      {animationDuration > 0 && (
        <>
          <Slider
            value={currentAnimationTime}
            size="small"
            onChange={(_, value) => {
              setAnimationSeekTime(value as number)
            }}
            sx={{
              color: "#a2c9f5",
              margin: "auto",
              marginTop: "1rem",
              width: "70%",
              "& .MuiSlider-markLabel": { color: "white" },
              "& .MuiSlider-track, & .MuiSlider-rail, & .MuiSlider-thumb": { transition: "none" },
            }}
            min={0}
            max={animationDuration}
            step={0.1}
            aria-label="Custom marks"
            marks={marks}
          />
          <IconButton
            onClick={() => setIsPlaying(!isPlaying)}
            sx={{ margin: "auto", marginTop: "-.5rem", color: "#a2c9f5", width: "2rem" }}
          >
            {isPlaying ? <Pause /> : <PlayArrow />}
          </IconButton>
        </>
      )}
    </FormControl>
  )
}

export default Animation
