"use client";

import { useEffect, useMemo, useState } from "react";
import {
  getHealth,
  getLocalF1Data,
  getMetrics,
  predictRace,
} from "@/lib/f1-ranker-api";
import { ensureGuestId, savePrediction } from "@/lib/supabase";
import type {
  LocalF1Data,
  ParticipantRequest,
  PredictionItem,
  PredictionRequest,
  Race,
  SavedPrediction,
} from "@/lib/types";

type Props = {
  mode: "free" | "race";
  selectedRace?: Race | null;
};

type AveragedPrediction = PredictionItem & {
  average_position: number;
  average_score: number;
  runs: number;
};

const emptyParticipant: ParticipantRequest = {
  driverId: 0,
  constructorId: 0,
  grid: null,
  qualifying_position: null,
  q1: null,
  q2: null,
  q3: null,
};

function toNullableNumber(value: FormDataEntryValue | null) {
  const raw = String(value ?? "").trim();
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function toNullableText(value: FormDataEntryValue | null) {
  const raw = String(value ?? "").trim();
  return raw ? raw : null;
}

function buildAverages(items: PredictionItem[][]): AveragedPrediction[] {
  const grouped = new Map<number, { position: number; score: number; runs: number }>();

  for (const run of items) {
    for (const item of run) {
      const current = grouped.get(item.driverId) ?? {
        position: 0,
        score: 0,
        runs: 0,
      };
      current.position += item.predicted_position;
      current.score += item.score;
      current.runs += 1;
      grouped.set(item.driverId, current);
    }
  }

  return Array.from(grouped.entries())
    .map(([driverId, value]) => ({
      driverId,
      predicted_position: 0,
      score: value.score / value.runs,
      average_position: value.position / value.runs,
      average_score: value.score / value.runs,
      runs: value.runs,
    }))
    .sort((a, b) => a.average_position - b.average_position)
    .map((item, index) => ({ ...item, predicted_position: index + 1 }));
}

function buildBlankRoster(data: LocalF1Data): ParticipantRequest[] {
  return data.drivers.map((driver) => ({
    driverId: driver.driverId,
    constructorId: driver.constructorId,
    grid: null,
    qualifying_position: null,
    q1: null,
    q2: null,
    q3: null,
  }));
}

export function RaceSimulator({ mode, selectedRace }: Props) {
  const [data, setData] = useState<LocalF1Data | null>(null);
  const [raceId, setRaceId] = useState(selectedRace ? String(selectedRace.raceId) : "");
  const [circuitId, setCircuitId] = useState(selectedRace ? String(selectedRace.circuitId) : "");
  const [raceDate, setRaceDate] = useState(selectedRace?.date ?? "");
  const [participants, setParticipants] = useState<ParticipantRequest[]>([
    emptyParticipant,
    emptyParticipant,
  ]);
  const [simulationCount, setSimulationCount] = useState(10);
  const [health, setHealth] = useState<"checking" | "ok" | "error">("checking");
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AveragedPrediction[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [savedMode, setSavedMode] = useState<string | null>(null);

  useEffect(() => {
    ensureGuestId();
    Promise.allSettled([getLocalF1Data(), getHealth(), getMetrics()]).then(
      ([localData, healthResult, metricsResult]) => {
        if (localData.status === "fulfilled") {
          setData(localData.value);
          const firstRace = selectedRace ?? localData.value.races.find((race) => race.status === "future");
          if (firstRace) {
            setRaceId(String(firstRace.raceId));
            setCircuitId(String(firstRace.circuitId));
            setRaceDate(firstRace.date);
          }
          setParticipants(
            mode === "race"
              ? buildBlankRoster(localData.value)
              : localData.value.latestParticipants.slice(0, 20),
          );
        }
        setHealth(healthResult.status === "fulfilled" ? "ok" : "error");
        if (metricsResult.status === "fulfilled") setMetrics(metricsResult.value);
      },
    );
  }, [mode, selectedRace]);

  const currentRace = useMemo(() => {
    if (!data) return null;
    return data.races.find((race) => race.raceId === Number(raceId)) ?? null;
  }, [data, raceId]);

  function getDriverLabel(driverId: number) {
    return data?.drivers.find((driver) => driver.driverId === driverId)?.label ?? `Piloto ${driverId}`;
  }

  function getDriverName(driverId: number) {
    return data?.drivers.find((driver) => driver.driverId === driverId)?.name ?? `Piloto ${driverId}`;
  }

  function getTeamName(constructorId: number) {
    return data?.constructors.find((team) => team.constructorId === constructorId)?.name ?? `Equipo ${constructorId}`;
  }

  function loadExample() {
    if (!data) return;
    const race = selectedRace ?? data.races.find((item) => item.status === "future") ?? data.races.at(-1);
    if (race) {
      setRaceId(String(race.raceId));
      setCircuitId(String(race.circuitId));
      setRaceDate(race.date);
    }
    setParticipants(
      mode === "race" ? buildBlankRoster(data) : data.latestParticipants.slice(0, 20),
    );
    setResult([]);
    setError(null);
  }

  function updateParticipant(index: number, next: ParticipantRequest) {
    setParticipants((current) =>
      current.map((item, itemIndex) => (itemIndex === index ? next : item)),
    );
  }

  function removeParticipant(index: number) {
    setParticipants((current) => current.filter((_, itemIndex) => itemIndex !== index));
  }

  async function submit() {
    setError(null);
    setSavedMode(null);
    setResult([]);

    const payload: PredictionRequest = {
      race_id: Number(raceId),
      circuit_id: Number(circuitId),
      race_date: raceDate || null,
      participants,
    };

    if (!payload.race_id || !payload.circuit_id) {
      setError("Debes seleccionar una carrera y un circuito validos.");
      return;
    }
    if (participants.length < 2 || participants.some((item) => !item.driverId || !item.constructorId)) {
      setError("Agrega al menos 2 pilotos con driverId y constructorId.");
      return;
    }

    setRunning(true);
    setProgress(0);

    try {
      const runs = [];
      const total = Math.min(Math.max(simulationCount, 1), 100);
      for (let index = 0; index < total; index += 1) {
        const prediction = await predictRace(payload);
        runs.push(prediction);
        setProgress(index + 1);
      }

      const averaged = buildAverages(runs.map((run) => run.predictions));
      setResult(averaged);

      const race = currentRace ?? {
        raceId: payload.race_id,
        circuitId: payload.circuit_id,
        name: "Prediccion libre",
        date: payload.race_date ?? new Date().toISOString().slice(0, 10),
      };
      const saved: SavedPrediction = {
        id: crypto.randomUUID(),
        created_at: new Date().toISOString(),
        source: mode === "race" ? "races" : "predicts",
        simulation_count: total,
        race,
        request: payload,
        averaged_predictions: averaged,
        raw_predictions: runs,
      };
      const saveResult = await savePrediction(saved);
      setSavedMode(saveResult.mode === "supabase" ? "Supabase" : "localStorage");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "No se pudo simular la carrera.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
      <section className="panel">
        <div className="flex flex-col gap-4 border-b border-slate-200 p-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold">
              {mode === "race" ? "Prediccion de carrera" : "Simulacion libre"}
            </h1>
            <p className="text-sm text-slate-500">
              Ejecuta hasta 100 predicciones contra la API local y guarda el promedio.
            </p>
          </div>
          <span className={`status ${health === "ok" ? "status-ok" : "status-error"}`}>
            API {health === "ok" ? "online" : health === "checking" ? "revisando" : "offline"}
          </span>
        </div>

        <div className="grid gap-4 p-4 md:grid-cols-4">
          <label className="field">
            Carrera
            <select value={raceId} onChange={(event) => {
              const race = data?.races.find((item) => item.raceId === Number(event.target.value));
              setRaceId(event.target.value);
              if (race) {
                setCircuitId(String(race.circuitId));
                setRaceDate(race.date);
              }
            }}>
              {data?.races.map((race) => (
                <option key={race.raceId} value={race.raceId}>
                  {race.year} R{race.round} - {race.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Circuito
            <select value={circuitId} onChange={(event) => setCircuitId(event.target.value)}>
              {data?.circuits.map((circuit) => (
                <option key={circuit.circuitId} value={circuit.circuitId}>
                  {circuit.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Fecha
            <input value={raceDate} type="date" onChange={(event) => setRaceDate(event.target.value)} />
          </label>
          <label className="field">
            Simulaciones
            <input
              max={100}
              min={1}
              type="number"
              value={simulationCount}
              onChange={(event) => setSimulationCount(Number(event.target.value))}
            />
          </label>
        </div>

        <div className="flex flex-wrap gap-2 px-4 pb-4">
          <button className="button-secondary" onClick={loadExample} type="button">
            Cargar ejemplo
          </button>
          <button
            className="button-secondary"
            onClick={() => setParticipants((current) => [...current, emptyParticipant])}
            type="button"
          >
            Agregar piloto
          </button>
          <button className="button-primary" disabled={running} onClick={submit} type="button">
            {running ? `Simulando ${progress}/${simulationCount}` : "Ejecutar simulacion"}
          </button>
        </div>

        {error ? <div className="mx-4 mb-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div> : null}

        <div className="overflow-x-auto border-t border-slate-200">
          <table className="data-table">
            <thead>
              <tr>
                <th>Piloto</th>
                <th>Equipo</th>
                <th>Grid</th>
                <th>Qualy</th>
                <th>Q1</th>
                <th>Q2</th>
                <th>Q3</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {participants.map((participant, index) => (
                <tr key={`${participant.driverId}-${index}`}>
                  <td>
                    <select
                      value={participant.driverId}
                      onChange={(event) => {
                        const driver = data?.drivers.find(
                          (item) => item.driverId === Number(event.target.value),
                        );
                        updateParticipant(index, {
                          ...participant,
                          driverId: Number(event.target.value),
                          constructorId: driver?.constructorId ?? participant.constructorId,
                        });
                      }}
                    >
                      <option value={0}>Seleccionar</option>
                      {data?.drivers.map((driver) => (
                        <option key={driver.driverId} value={driver.driverId}>
                          {driver.label}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <select
                      value={participant.constructorId}
                      onChange={(event) =>
                        updateParticipant(index, {
                          ...participant,
                          constructorId: Number(event.target.value),
                        })
                      }
                    >
                      <option value={0}>Seleccionar</option>
                      {data?.constructors.map((team) => (
                        <option key={team.constructorId} value={team.constructorId}>
                          {team.label}
                        </option>
                      ))}
                    </select>
                    <span className="mt-1 block text-xs text-[var(--muted)]">
                      {getTeamName(participant.constructorId)}
                    </span>
                  </td>
                  <td><input value={participant.grid ?? ""} onChange={(event) => updateParticipant(index, { ...participant, grid: toNullableNumber(event.target.value) })} /></td>
                  <td><input value={participant.qualifying_position ?? ""} onChange={(event) => updateParticipant(index, { ...participant, qualifying_position: toNullableNumber(event.target.value) })} /></td>
                  <td><input value={participant.q1 ?? ""} onChange={(event) => updateParticipant(index, { ...participant, q1: toNullableText(event.target.value) })} /></td>
                  <td><input value={participant.q2 ?? ""} onChange={(event) => updateParticipant(index, { ...participant, q2: toNullableText(event.target.value) })} /></td>
                  <td><input value={participant.q3 ?? ""} onChange={(event) => updateParticipant(index, { ...participant, q3: toNullableText(event.target.value) })} /></td>
                  <td className="text-right">
                    <button className="text-sm text-red-600" onClick={() => removeParticipant(index)} type="button">
                      Quitar
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <aside className="space-y-6">
        <section className="panel p-4">
          <h2 className="text-lg font-semibold">Resultado promedio</h2>
          {savedMode ? <p className="mt-1 text-sm text-slate-500">Guardado en {savedMode}.</p> : null}
          <div className="mt-4 overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr><th>Pos</th><th>Piloto</th><th>Score</th></tr>
              </thead>
              <tbody>
                {result.map((item) => (
                  <tr key={item.driverId}>
                    <td>{item.predicted_position}</td>
                    <td>
                      <div className="font-medium">{getDriverName(item.driverId)}</div>
                      <div className="text-xs text-[var(--muted)]">{getDriverLabel(item.driverId)}</div>
                    </td>
                    <td>{item.average_score.toFixed(4)}</td>
                  </tr>
                ))}
                {!result.length ? <tr><td colSpan={3}>Sin resultados aun.</td></tr> : null}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel p-4">
          <h2 className="text-lg font-semibold">Metricas modelo</h2>
          <pre className="mt-3 max-h-64 overflow-auto rounded-md bg-slate-950 p-3 text-xs text-white">
            {metrics ? JSON.stringify(metrics, null, 2) : "Sin metricas cargadas"}
          </pre>
        </section>
      </aside>
    </div>
  );
}
