import React, { useState } from "react"
import { List, ListItem, ListItemText, Collapse, IconButton, Slider, Typography, ListItemButton } from "@mui/material"
import { ExpandLess, ExpandMore } from "@mui/icons-material"

const importantBones = [
  { name_jp: "センター", name_en: "center" },
  { name_jp: "首", name_en: "neck" },
  { name_jp: "頭", name_en: "head" },
  { name_jp: "上半身", name_en: "upperBody" },
  { name_jp: "下半身", name_en: "lowerBody" },
  { name_jp: "左足", name_en: "leftLeg" },
  { name_jp: "右足", name_en: "rightLeg" },
  { name_jp: "左ひざ", name_en: "leftKnee" },
  { name_jp: "右ひざ", name_en: "rightKnee" },
  { name_jp: "左足首", name_en: "leftAnkle" },
  { name_jp: "右足首", name_en: "rightAnkle" },
  { name_jp: "左腕", name_en: "leftForearm" },
  { name_jp: "右腕", name_en: "rightForearm" },
  { name_jp: "左ひじ", name_en: "leftElbow" },
  { name_jp: "右ひじ", name_en: "rightElbow" },
  { name_jp: "左手首", name_en: "leftWrist" },
  { name_jp: "右手首", name_en: "rightWrist" },
  { name_jp: "左足ＩＫ", name_en: "leftFootIK" },
  { name_jp: "右足ＩＫ", name_en: "rightFootIK" },
  { name_jp: "左目", name_en: "leftEye" },
  { name_jp: "右目", name_en: "rightEye" },
  { name_jp: "右親指１", name_en: "rightThumb1" },
  { name_jp: "右親指２", name_en: "rightThumb2" },
  { name_jp: "右人指１", name_en: "rightIndex1" },
  { name_jp: "右人指２", name_en: "rightIndex2" },
  { name_jp: "右人指３", name_en: "rightIndex3" },
  { name_jp: "右中指１", name_en: "rightMiddle1" },
  { name_jp: "右中指２", name_en: "rightMiddle2" },
  { name_jp: "右中指３", name_en: "rightMiddle3" },
  { name_jp: "右薬指１", name_en: "rightRing1" },
  { name_jp: "右薬指２", name_en: "rightRing2" },
  { name_jp: "右薬指３", name_en: "rightRing3" },
  { name_jp: "右小指１", name_en: "rightPinky1" },
  { name_jp: "右小指２", name_en: "rightPinky2" },
  { name_jp: "右小指３", name_en: "rightPinky3" },
  { name_jp: "左親指１", name_en: "leftThumb1" },
  { name_jp: "左親指２", name_en: "leftThumb2" },
  { name_jp: "左人指１", name_en: "leftIndex1" },
  { name_jp: "左人指２", name_en: "leftIndex2" },
  { name_jp: "左人指３", name_en: "leftIndex3" },
  { name_jp: "左中指１", name_en: "leftMiddle1" },
  { name_jp: "左中指２", name_en: "leftMiddle2" },
  { name_jp: "左中指３", name_en: "leftMiddle3" },
  { name_jp: "左薬指１", name_en: "leftRing1" },
  { name_jp: "左薬指２", name_en: "leftRing2" },
  { name_jp: "左薬指３", name_en: "leftRing3" },
  { name_jp: "左小指１", name_en: "leftPinky1" },
  { name_jp: "左小指２", name_en: "leftPinky2" },
  { name_jp: "左小指３", name_en: "leftPinky3" },
]

const categories = {
  Body: ["center", "neck", "head", "upperBody", "lowerBody"],
  Legs: ["leftLeg", "rightLeg", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle", "leftFootIK", "rightFootIK"],
  Arms: ["leftForearm", "rightForearm", "leftElbow", "rightElbow", "leftWrist", "rightWrist"],
  Eyes: ["leftEye", "rightEye"],
  Fingers: [
    "rightThumb1",
    "rightThumb2",
    "rightIndex1",
    "rightIndex2",
    "rightIndex3",
    "rightMiddle1",
    "rightMiddle2",
    "rightMiddle3",
    "rightRing1",
    "rightRing2",
    "rightRing3",
    "rightPinky1",
    "rightPinky2",
    "rightPinky3",
    "leftThumb1",
    "leftThumb2",
    "leftIndex1",
    "leftIndex2",
    "leftIndex3",
    "leftMiddle1",
    "leftMiddle2",
    "leftMiddle3",
    "leftRing1",
    "leftRing2",
    "leftRing3",
    "leftPinky1",
    "leftPinky2",
    "leftPinky3",
  ],
}

function Skeleton({
  setBoneRotation,
}: {
  setBoneRotation: (boneRotation: { name: string; axis: string; value: number }) => void
}): JSX.Element {
  const [openCategory, setOpenCategory] = useState<string | null>(null)
  const [openBones, setOpenBones] = useState<Record<string, boolean>>({})

  const toggleCategory = (category: string) => {
    setOpenCategory((prev) => (prev === category ? null : category))
  }

  const toggleBone = (bone: string) => {
    setOpenBones((prev) => ({ ...prev, [bone]: !prev[bone] }))
  }

  return (
    <List className="skeleton" dense>
      {Object.entries(categories).map(([category, bones]) => (
        <React.Fragment key={category}>
          <ListItemButton onClick={() => toggleCategory(category)}>
            <ListItemText primary={category} />
            <IconButton edge="end">
              {openCategory === category ? (
                <ExpandLess sx={{ color: "white" }} />
              ) : (
                <ExpandMore sx={{ color: "white" }} />
              )}
            </IconButton>
          </ListItemButton>
          <Collapse in={openCategory === category} timeout="auto" unmountOnExit>
            <List component="div" disablePadding dense>
              {bones.map((boneName) => {
                const bone = importantBones.find((b) => b.name_en === boneName)
                return bone ? (
                  <React.Fragment key={bone.name_en}>
                    <ListItemButton onClick={() => toggleBone(bone.name_en)} sx={{ pl: 4 }}>
                      <ListItemText primary={`${bone.name_jp} - ${bone.name_en}`} />
                      <IconButton edge="end">
                        {openBones[bone.name_en] ? (
                          <ExpandLess sx={{ color: "white" }} />
                        ) : (
                          <ExpandMore sx={{ color: "white" }} />
                        )}
                      </IconButton>
                    </ListItemButton>
                    <Collapse in={openBones[bone.name_en]} timeout="auto" unmountOnExit>
                      <List component="div" disablePadding dense>
                        {["x", "y", "z"].map((axis) => (
                          <ListItem key={`${bone.name_en}-${axis}`} sx={{ pl: 6 }}>
                            <Typography variant="body2" style={{ minWidth: "20px" }}>
                              {axis}:
                            </Typography>
                            <Slider
                              size="small"
                              defaultValue={0}
                              onChange={(_, newValue) => {
                                setBoneRotation({
                                  name: bone.name_jp,
                                  axis: axis,
                                  value: newValue as number,
                                })
                              }}
                              aria-label={`${axis} axis`}
                              valueLabelDisplay="auto"
                              valueLabelFormat={(value) => `${((value * 180) / Math.PI).toFixed(0)}°`}
                              step={0.001}
                              min={-Math.PI}
                              max={Math.PI}
                              sx={{ ml: 2, width: "80%", color: "#a2c9f5" }}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Collapse>
                  </React.Fragment>
                ) : null
              })}
            </List>
          </Collapse>
        </React.Fragment>
      ))}
    </List>
  )
}

export default Skeleton
