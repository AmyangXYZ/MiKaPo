import { Pause, PlayArrow } from "@mui/icons-material"
import { FormControl, FormControlLabel, IconButton, Radio, RadioGroup, Slider, Typography } from "@mui/material"
import { useMemo } from "react"

const availableAnimations = ["Stand", "Zyy", "Miku", "iKun1", "0-540"]

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
      { value: animationDuration, label: "-" + formatTime(animationDuration - currentAnimationTime) },
    ]
  }, [currentAnimationTime, animationDuration])

  return (
    <FormControl className="animation">
      <RadioGroup
        aria-labelledby="demo-radio-buttons-group-label"
        defaultValue=""
        name="radio-buttons-group"
        onChange={(_, value) => setSelectedAnimation(value)}
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
