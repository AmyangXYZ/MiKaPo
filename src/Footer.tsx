import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faBone, faFilm, faPanorama, faShirt, faUser } from "@fortawesome/free-solid-svg-icons"
import { Fab, Tooltip } from "@mui/material"
import { SportsMartialArts } from "@mui/icons-material"

function Footer({
  setOpenDrawer,
  setActiveTab,
}: {
  setOpenDrawer: (open: boolean) => void
  setActiveTab: (tab: string) => void
}): JSX.Element {
  const colorPalette = {
    motion: "#4A90E2", // Soft Blue
    skeleton: "#3498DB", // Peter River Blue
    material: "#2ECC71", // Emerald Green
    background: "#9B59B6", // Amethyst
    model: "#FF8C00", // Dark Orange
    animation: "#E74C3C", // Alizarin Red
  }

  return (
    <div className="footer">
      <Tooltip title="Motion capture" placement="left-start">
        <Fab
          style={{
            position: "absolute",
            right: 20,
            bottom: 20,
            width: "106px",
            height: "106px",
            backgroundColor: colorPalette.motion,
          }}
          onClick={() => {
            setActiveTab("motion")
            setOpenDrawer(true)
          }}
        >
          <SportsMartialArts sx={{ fontSize: "96px", color: "white", marginTop: "10px" }} />
        </Fab>
      </Tooltip>
      {[
        { name: "Model", icon: faUser, angle: -20, color: colorPalette.model },
        { name: "Material", icon: faShirt, angle: 10, color: colorPalette.material },
        { name: "Background", icon: faPanorama, angle: 40, color: colorPalette.background },
        { name: "Skeleton", icon: faBone, angle: 70, color: colorPalette.skeleton },
        { name: "Animation", icon: faFilm, angle: 100, color: colorPalette.animation },
      ].map(({ name, icon, angle, color }, index) => (
        <Tooltip key={index} title={name}>
          <Fab
            style={{
              position: "absolute",
              right: 50 + 90 * Math.cos((angle * Math.PI) / 180),
              bottom: 50 + 90 * Math.sin((angle * Math.PI) / 180),
              width: "36px",
              height: "36px",
              backgroundColor: color,
            }}
            onClick={() => {
              setActiveTab(name.toLowerCase())
              setOpenDrawer(true)
            }}
          >
            <FontAwesomeIcon icon={icon} color="white" size="lg" />
          </Fab>
        </Tooltip>
      ))}
    </div>
  )
}

export default Footer
