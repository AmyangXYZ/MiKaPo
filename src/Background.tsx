import { Box, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent } from "@mui/material"

const availableBackgrounds = ["Static", "Office", "Beach", "Bedroom"]

function Background({
  selectedBackground,
  setSelectedBackground,
  style,
}: {
  selectedBackground: string
  setSelectedBackground: (background: string) => void
  style: React.CSSProperties
}): JSX.Element {
  const handleBackgroundChange = (event: SelectChangeEvent): void => {
    setSelectedBackground(event.target.value)
  }

  return (
    <div className="background" style={style}>
      <Box className="background-selector">
        <FormControl sx={{ borderColor: "white" }}>
          <InputLabel sx={{ color: "white", fontSize: ".9rem" }}>Background</InputLabel>
          <Select
            label="Background"
            value={selectedBackground}
            onChange={handleBackgroundChange}
            sx={{ color: "white", fontSize: ".9rem" }}
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
