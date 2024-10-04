import { FormControl, FormControlLabel, Radio, RadioGroup, Avatar, Typography } from "@mui/material"
import { Box } from "@mui/system"

const availableModels = ["深空之眼-托特", "深空之眼-托特2", "深空之眼-梵天", "鸣潮-吟霖", "原神-荧"]

function Model({ setSelectedModel }: { setSelectedModel: (model: string) => void }): JSX.Element {
  return (
    <FormControl className="model">
      <RadioGroup
        aria-labelledby="demo-radio-buttons-group-label"
        defaultValue=""
        name="radio-buttons-group"
        onChange={(e) => setSelectedModel(e.target.value)}
        sx={{ display: "flex", margin: "auto" }}
      >
        {availableModels.map((model) => (
          <FormControlLabel
            key={model}
            value={model}
            control={
              <Radio
                sx={{ color: "#a2c9f5", "&.Mui-checked": { color: "#a2c9f5" }, marginLeft: 2, marginBottom: 2 }}
                size="small"
              />
            }
            label={
              <Box sx={{ display: "flex", alignItems: "center", marginBottom: 2 }}>
                <Avatar src={`/avatar/${model}.png`} alt={model} sx={{ width: 64, height: 64, marginRight: 1 }} />
                <Typography sx={{ fontSize: ".9rem" }}>{model}</Typography>
              </Box>
            }
          />
        ))}
      </RadioGroup>
    </FormControl>
  )
}

export default Model
