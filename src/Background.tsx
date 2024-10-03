import { Box, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent } from "@mui/material"

const availableBackgrounds = ["Static", "Office", "Beach", "Bedroom"]

function Background({
  selectedBackground,
  setSelectedBackground,
}: {
  selectedBackground: string
  setSelectedBackground: (background: string) => void
}): JSX.Element {
  const handleBackgroundChange = (event: SelectChangeEvent): void => {
    setSelectedBackground(event.target.value)
  }

  return (
    <div className="background">
      <Box className="background-selector">
        <FormControl sx={{ borderColor: "white" }}>
          <InputLabel sx={{ color: "white", fontSize: ".9rem" }}>Background</InputLabel>
          <Select
            label="Background"
            value={selectedBackground}
            onChange={handleBackgroundChange}
            sx={{
              color: "white",
              fontSize: ".9rem",
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
            {availableBackgrounds.map((background) => (
              <MenuItem sx={{ fontSize: ".8rem" }} key={background} value={background}>
                {background}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
    </div>
  )
}

export default Background
