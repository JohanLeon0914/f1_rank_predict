import { RaceSimulator } from "@/components/RaceSimulator";

export default function PredictsPage() {
  return (
    <section className="page-shell mx-auto max-w-7xl px-4 pb-8">
      <RaceSimulator mode="free" />
    </section>
  );
}
