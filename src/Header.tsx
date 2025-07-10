import { Avatar, IconButton } from "@mui/material"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faGithub } from "@fortawesome/free-brands-svg-icons"
import { Download } from "@mui/icons-material"

function Header(): JSX.Element {
  return (
    <header className="header">
      <div className="header-item" style={{ justifyContent: "flex-start" }}>
        <Avatar
          alt="MiKaPo"
          src="/logo.png"
          sx={{
            width: 36,
            height: 36,
            marginRight: ".5rem",
            transition: "transform 2s ease-in-out",
            "&:hover": {
              transform: "rotate(360deg)",
            },
          }}
        />
        <h2 className="header-item-mobile-hidden">MiKaPo</h2>
      </div>

      <div className="header-item" style={{ justifyContent: "flex-end" }}>
        <a href="https://github.com/AmyangXYZ/MiKaPo" target="_blank">
          <IconButton>
            <FontAwesomeIcon icon={faGithub} color="white" size="sm" />
          </IconButton>
        </a>
        <a href="https://github.com/AmyangXYZ/MiKaPo-Electron" target="_blank" className="header-item-mobile-hidden">
          <IconButton size="small" color="inherit">
            <Download sx={{ color: "white", fontSize: "1.5rem", marginTop: ".2rem" }} />
          </IconButton>
        </a>
        <a href="https://popo.love" target="_blank">
          <div
            style={{
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              color: "white",
              padding: "6px 12px",
              borderRadius: "20px",
              fontSize: "12px",
              fontWeight: "600",
              textDecoration: "none",
              transition: "all 0.3s ease",
              cursor: "pointer",
              display: "inline-block",
            }}
          >
            <span className="header-item-mobile-hidden">🤖 Try my new project: Pose MMD with LLM</span>
            <span className="header-item-mobile-only">🤖 PoPo: AI Pose Gen</span>
          </div>
        </a>
      </div>
    </header>
  )
}

export default Header
