interface HeaderProps {
  fps: number
}

function Header({ fps }: HeaderProps): JSX.Element {
  return (
    <header className="header">
      <a href="https://github.com/AmyangXYZ/MiKaPo" target="_blank">
        <h2>MiKaPo</h2>
      </a>
      <p>FPS: {fps}</p>
      <a href="https://github.com/AmyangXYZ/MiKaPo-electron" target="_blank">
        <h4>Download</h4>
      </a>
    </header>
  )
}

export default Header
