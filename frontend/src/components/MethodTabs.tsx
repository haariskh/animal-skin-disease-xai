import type { Explanation } from "../types";

interface MethodTabsProps {
  explanations: Explanation[];
  active: string;
  onChange: (method: string) => void;
}

export default function MethodTabs({ explanations, active, onChange }: MethodTabsProps) {
  const current = explanations.find((item) => item.method === active);

  return (
    <div>
      <div className="methods" role="tablist" aria-label="Explanation method">
        {explanations.map((item) => (
          <button
            key={item.method}
            type="button"
            role="tab"
            aria-selected={item.method === active}
            className="methods__tab"
            onClick={() => onChange(item.method)}
          >
            {item.method_label}
          </button>
        ))}
      </div>
      {current && <p className="methods__note">{current.description}</p>}
    </div>
  );
}
