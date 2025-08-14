import { Button } from "./ui/button"
import { X } from "lucide-react"
import { ScrollArea } from "./ui/scroll-area"
import { RadioGroup, RadioGroupItem } from "./ui/radio-group"
import { Label } from "./ui/label"

export default function ModelsPanel({
  open,
  setOpen,
  selectedModel,
  selectModel,
}: {
  open: boolean
  setOpen: (open: boolean) => void
  selectedModel: string
  selectModel: (model: string) => void
}) {

  const models = ["深空之眼-塞勒涅", "深空之眼-梵天"]

  return (
    <div
      className={`fixed right-0 top-0 h-full w-60 bg-background border-l shadow-lg z-250 flex flex-col transition-transform duration-300 ease-in-out ${open ? "translate-x-0" : "translate-x-full"
        }`}
    >
      <div className="flex flex-col gap-1.5 p-4 border-b">
        <div className="flex items-center justify-between">
          <h4 className="scroll-m-20 text-base font-semibold tracking-tight">Models</h4>
          <Button size="icon" variant="ghost" onClick={() => setOpen(false)}>
            <X className="size-4" />
          </Button>
        </div>
      </div>

      <ScrollArea className="max-h-[calc(100dvh-22rem)] overflow-auto p-4 px-8">
        <RadioGroup defaultValue={selectedModel} onValueChange={selectModel} className="flex flex-col gap-4">
          {models.map((model) => (
            <div key={model} className="flex justify-between gap-3">
              <Label htmlFor={model}>{model}</Label>
              <RadioGroupItem value={model} id={model} />
            </div>
          ))}
        </RadioGroup>
      </ScrollArea>
    </div>
  )
}
