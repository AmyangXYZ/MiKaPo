import { Box, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent } from "@mui/material"

const availableAnimations = ["Stand", "Zyy", "Miku", "iKun1", "0-540"]

function Animation({
  selectedAnimation,
  setSelectedAnimation,
}: {
  selectedAnimation: string
  setSelectedAnimation: (animation: string) => void
}): JSX.Element {
  const handleAnimationChange = (event: SelectChangeEvent): void => {
    setSelectedAnimation(event.target.value)
  }

  return (
    <div className="animation">
      <Box className="model-selector">
        <FormControl sx={{ borderColor: "white" }}>
          <InputLabel sx={{ color: "white", fontSize: ".9rem" }}>Animation</InputLabel>
          <Select
            label="Animation"
            value={selectedAnimation}
            onChange={handleAnimationChange}
            sx={{
              color: "white",
              fontSize: ".9rem",
              minWidth: "100px",
              "& .MuiOutlinedInput-notchedOutline": {
                borderColor: "white",
              },
              "&:hover .MuiOutlinedInput-notchedOutline": {
                borderColor: "lightgray",
              },
              "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                borderColor: "cyan",
              },
            }}
            autoWidth
          >
            {availableAnimations.map((animation) => (
              <MenuItem sx={{ fontSize: ".8rem" }} key={animation} value={animation}>
                {animation}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
    </div>
  )
}

export default Animation
