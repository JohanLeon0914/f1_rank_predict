import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import type {
  Circuit,
  ConstructorOption,
  DriverOption,
  LocalF1Data,
  ParticipantRequest,
  Race,
} from "@/lib/types";

const csvRoot = path.resolve(process.cwd(), "..", "F1", "CSVs");

const knownModernConstructorsById = new Map<number, string>([
  [1, "McLaren"],
  [3, "Williams"],
  [6, "Ferrari"],
  [9, "Red Bull"],
  [117, "Aston Martin"],
  [131, "Mercedes"],
  [210, "Haas F1 Team"],
  [214, "Alpine F1 Team"],
  [215, "RB F1 Team"],
  [216, "Cadillac F1 Team"],
  [217, "Audi"],
]);

type OpenF1Driver = {
  driver_number: number;
  full_name?: string;
  name_acronym?: string;
  team_name?: string;
  team_colour?: string;
};

type JolpicaDriver = {
  driverId: string;
  permanentNumber?: string;
  givenName?: string;
  familyName?: string;
};

type JolpicaDriverInfo = {
  fullName: string;
  teamName: string | null;
};

const knownModernDriversByNumber = new Map<number, JolpicaDriverInfo>([
  [6, { fullName: "Isack Hadjar", teamName: "Red Bull" }],
]);

const knownModernDriversById = new Map<
  number,
  { name: string; constructorId: number; number: string }
>([
  [846, { name: "Lando Norris", constructorId: 1, number: "1" }],
  [857, { name: "Oscar Piastri", constructorId: 1, number: "81" }],
  [830, { name: "Max Verstappen", constructorId: 9, number: "3" }],
  [865, { name: "Isack Hadjar", constructorId: 9, number: "6" }],
  [863, { name: "Andrea Kimi Antonelli", constructorId: 131, number: "12" }],
  [847, { name: "George Russell", constructorId: 131, number: "63" }],
  [844, { name: "Charles Leclerc", constructorId: 6, number: "16" }],
  [1, { name: "Lewis Hamilton", constructorId: 6, number: "44" }],
  [859, { name: "Liam Lawson", constructorId: 215, number: "30" }],
  [866, { name: "Arvid Lindblad", constructorId: 215, number: "41" }],
  [807, { name: "Nico Hulkenberg", constructorId: 217, number: "27" }],
  [864, { name: "Gabriel Bortoleto", constructorId: 217, number: "5" }],
  [842, { name: "Pierre Gasly", constructorId: 214, number: "10" }],
  [861, { name: "Franco Colapinto", constructorId: 214, number: "43" }],
  [840, { name: "Lance Stroll", constructorId: 117, number: "18" }],
  [4, { name: "Fernando Alonso", constructorId: 117, number: "14" }],
  [839, { name: "Esteban Ocon", constructorId: 210, number: "31" }],
  [860, { name: "Oliver Bearman", constructorId: 210, number: "87" }],
  [848, { name: "Alexander Albon", constructorId: 3, number: "23" }],
  [832, { name: "Carlos Sainz", constructorId: 3, number: "55" }],
  [815, { name: "Sergio Perez", constructorId: 216, number: "11" }],
  [822, { name: "Valtteri Bottas", constructorId: 216, number: "77" }],
]);

function parseCsv(text: string) {
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (char === '"' && quoted && next === '"') {
      field += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(field);
      if (row.some(Boolean)) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }

  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }

  const [headers, ...values] = rows;
  return values.map((items) =>
    Object.fromEntries(headers.map((header, index) => [header, clean(items[index])])),
  );
}

function clean(value?: string) {
  if (!value || value === "\\N") return null;
  return value;
}

function numberOrNull(value: unknown) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

async function readCsv(fileName: string) {
  const text = await readFile(path.join(csvRoot, fileName), "utf8");
  return parseCsv(text);
}

async function readOpenF1Drivers() {
  try {
    const response = await fetch(
      "https://api.openf1.org/v1/drivers?session_key=latest",
      { cache: "no-store", signal: AbortSignal.timeout(2500) },
    );
    if (!response.ok) return new Map<number, OpenF1Driver>();
    const drivers = (await response.json()) as OpenF1Driver[];
    return new Map(drivers.map((driver) => [driver.driver_number, driver]));
  } catch {
    return new Map<number, OpenF1Driver>();
  }
}

async function readJolpicaDriverInfo(year: number) {
  try {
    const response = await fetch(
      `https://api.jolpi.ca/ergast/f1/${year}/drivers.json?limit=100`,
      {
        headers: { "User-Agent": "F1MLPredicts/0.1.0 NextJS" },
        cache: "no-store",
        signal: AbortSignal.timeout(2500),
      },
    );
    if (!response.ok) return new Map<number, JolpicaDriverInfo>();

    const payload = await response.json();
    const drivers = (payload.MRData?.DriverTable?.Drivers ?? []) as JolpicaDriver[];

    return new Map(
      drivers
        .filter((driver) => Number.isFinite(Number(driver.permanentNumber)))
        .map((driver) => [
          Number(driver.permanentNumber),
          {
            fullName: `${driver.givenName ?? ""} ${driver.familyName ?? ""}`.trim(),
            teamName: knownModernDriversByNumber.get(Number(driver.permanentNumber))?.teamName ?? null,
          },
        ]),
    );
  } catch {
    return new Map<number, JolpicaDriverInfo>();
  }
}

export async function GET() {
  try {
    const [raceRows, circuitRows, resultRows, qualifyingRows, openF1ByNumber] =
      await Promise.all([
        readCsv("races.csv"),
        readCsv("circuits.csv"),
        readCsv("results.csv"),
        readCsv("qualifying.csv"),
        readOpenF1Drivers(),
      ]);

    const today = new Date().toISOString().slice(0, 10);
    const circuits: Circuit[] = circuitRows.map((row) => ({
      circuitId: Number(row.circuitId),
      circuitRef: String(row.circuitRef),
      name: String(row.name),
      location: String(row.location),
      country: String(row.country),
    }));
    const circuitById = new Map(circuits.map((circuit) => [circuit.circuitId, circuit]));

    const races: Race[] = raceRows
      .map((row) => {
        const race: Race = {
          raceId: Number(row.raceId),
          year: Number(row.year),
          round: Number(row.round),
          circuitId: Number(row.circuitId),
          name: String(row.name),
          date: String(row.date),
          time: row.time ? String(row.time) : null,
          status: String(row.date) >= today ? "future" : "past",
          circuit: circuitById.get(Number(row.circuitId)),
        };
        return race;
      })
      .sort((a, b) => a.date.localeCompare(b.date));

    const latestDatasetYear = Math.max(
      ...races.map((race) => race.year).filter(Number.isFinite),
    );
    const jolpicaByNumber = await readJolpicaDriverInfo(latestDatasetYear);

    const latestRaceId = Math.max(
      ...resultRows.map((row) => Number(row.raceId)).filter(Number.isFinite),
    );
    const latestResults = resultRows.filter(
      (row) => Number(row.raceId) === latestRaceId,
    );
    const latestQualifyingByDriver = new Map(
      qualifyingRows
        .filter((row) => Number(row.raceId) === latestRaceId)
        .map((row) => [Number(row.driverId), row]),
    );

    const latestParticipants: ParticipantRequest[] = latestResults
      .map((row) => {
        const qualifying = latestQualifyingByDriver.get(Number(row.driverId));
        return {
          driverId: Number(row.driverId),
          constructorId: Number(row.constructorId),
          grid: numberOrNull(row.grid),
          qualifying_position: numberOrNull(qualifying?.position),
          q1: qualifying?.q1 ? String(qualifying.q1) : null,
          q2: qualifying?.q2 ? String(qualifying.q2) : null,
          q3: qualifying?.q3 ? String(qualifying.q3) : null,
        };
      })
      .sort((a, b) => (a.grid ?? 99) - (b.grid ?? 99));

    const drivers: DriverOption[] = latestResults
      .map((row) => ({
        driverId: Number(row.driverId),
        number: row.number ? String(row.number) : null,
        constructorId: Number(row.constructorId),
        known: knownModernDriversById.get(Number(row.driverId)),
        openF1: row.number ? openF1ByNumber.get(Number(row.number)) : undefined,
        jolpica: row.number
          ? jolpicaByNumber.get(Number(row.number)) ??
            knownModernDriversByNumber.get(Number(row.number))
          : undefined,
      }))
      .map((driver) => {
        const name =
          driver.known?.name ??
          driver.jolpica?.fullName ??
          driver.openF1?.full_name ??
          `Piloto ${driver.driverId}`;
        const teamName =
          knownModernConstructorsById.get(driver.known?.constructorId ?? driver.constructorId) ??
          driver.jolpica?.teamName ??
          driver.openF1?.team_name ??
          `Equipo ${driver.constructorId}`;
        const number = driver.known?.number ?? driver.number;
        return {
          driverId: driver.driverId,
          number,
          constructorId: driver.known?.constructorId ?? driver.constructorId,
          name,
          teamName,
          teamColor: driver.openF1?.team_colour ?? null,
          label: `${name}${number ? ` #${number}` : ""} - ${teamName}`,
        };
      })
      .sort((a, b) => a.name.localeCompare(b.name));

    const constructorNames = new Map<number, string>();
    for (const driver of drivers) {
      if (!constructorNames.has(driver.constructorId)) {
        constructorNames.set(
          driver.constructorId,
          knownModernConstructorsById.get(driver.constructorId) ?? driver.teamName,
        );
      }
    }

    const constructors: ConstructorOption[] = Array.from(
      new Set(drivers.map((driver) => driver.constructorId).filter(Number.isFinite)),
    )
      .sort((a, b) => a - b)
      .map((constructorId) => ({
        constructorId,
        name:
          constructorNames.get(constructorId) ??
          knownModernConstructorsById.get(constructorId) ??
          `Equipo ${constructorId}`,
        label:
          constructorNames.get(constructorId) ??
          knownModernConstructorsById.get(constructorId) ??
          `Equipo ${constructorId}`,
      }));

    const data: LocalF1Data = {
      races,
      circuits,
      drivers,
      constructors,
      latestParticipants,
    };

    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      {
        detail:
          error instanceof Error
            ? error.message
            : "No se pudieron leer los CSV locales.",
      },
      { status: 500 },
    );
  }
}
