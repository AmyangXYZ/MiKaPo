import { FormControl, FormControlLabel, Radio, RadioGroup, Typography } from "@mui/material"

const availableAnimations = ["Stand", "Zyy", "Miku", "iKun1", "0-540"]

function Animation({ setSelectedAnimation }: { setSelectedAnimation: (animation: string) => void }): JSX.Element {
  return (
    <FormControl className="animation">
      <RadioGroup
        aria-labelledby="demo-radio-buttons-group-label"
        defaultValue=""
        name="radio-buttons-group"
        onChange={(e) => setSelectedAnimation(e.target.value)}
        sx={{ display: "flex", margin: "auto" }}
      >
        {availableAnimations.map((animation) => (
          <FormControlLabel
            value={animation}
            control={<Radio sx={{ color: "lightgray" }} size="small" />}
            label={<Typography sx={{ fontSize: ".9rem" }}>{animation}</Typography>}
          />
        ))}
      </RadioGroup>
    </FormControl>
  )
}

export default Animation
