import "./assets/main.css"
import "@fontsource/roboto/300.css"
import "@fontsource/roboto/400.css"
import "@fontsource/roboto/500.css"
import "@fontsource/roboto/700.css"
import { Analytics } from "@vercel/analytics/react"

import ReactDOM from "react-dom/client"
import App from "./App"

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <>
    <App />
    <Analytics />
  </>
)
