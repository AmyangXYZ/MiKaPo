import { Checkbox, List, ListItem, Typography } from "@mui/material"

function Materials({
  materials,
  setMaterialVisible,
}: {
  materials: string[]
  setMaterialVisible: (material: { name: string; visible: boolean }) => void
}): JSX.Element {
  return (
    <List className="material" dense>
      {materials.map((material) => (
        <ListItem
          key={material}
          secondaryAction={
            <Checkbox
              defaultChecked
              onChange={(e) => {
                setMaterialVisible({ name: material, visible: e.target.checked })
              }}
              size="small"
              sx={{
                color: "#a2c9f5",
                "&.Mui-checked": {
                  color: "#a2c9f5",
                },
              }}
            />
          }
        >
          <Typography>{material}</Typography>
        </ListItem>
      ))}
    </List>
  )
}

export default Materials
