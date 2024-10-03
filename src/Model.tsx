import { Box, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent } from "@mui/material"

const availableModels = ["深空之眼-托特", "深空之眼-托特2", "深空之眼-大梵天", "鸣潮-吟霖", "原神-荧"]

function Model({
  selectedModel,
  setSelectedModel,
}: {
  selectedModel: string
  setSelectedModel: (model: string) => void
}): JSX.Element {
  const handleModelChange = (event: SelectChangeEvent): void => {
    setSelectedModel(event.target.value)
  }

  return (
    <div className="model">
      <Box className="model-selector">
        <FormControl sx={{ borderColor: "white" }}>
          <InputLabel sx={{ color: "white", fontSize: ".9rem" }}>Model</InputLabel>
          <Select
            label="Model"
            value={selectedModel}
            onChange={handleModelChange}
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
            {availableModels.map((model) => (
              <MenuItem sx={{ fontSize: ".8rem" }} key={model} value={model}>
                {model}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
    </div>
  )
}

export default Model
