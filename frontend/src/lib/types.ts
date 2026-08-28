export type ParticipantRequest = {
  driverId: number;
  constructorId: number;
  grid: number | null;
  qualifying_position: number | null;
  q1: string | number | null;
  q2: string | number | null;
  q3: string | number | null;
};

export type PredictionRequest = {
  race_id: number;
  circuit_id: number;
  race_date?: string | null;
  participants: ParticipantRequest[];
};

export type PredictionItem = {
  predicted_position: number;
  driverId: number;
  score: number;
};

export type PredictionResponse = {
  race_id: number;
  circuit_id: number;
  predictions: PredictionItem[];
};

export type ApiHealth = {
  status: string;
};

export type Circuit = {
  circuitId: number;
  circuitRef: string;
  name: string;
  location: string;
  country: string;
};

export type Race = {
  raceId: number;
  year: number;
  round: number;
  circuitId: number;
  name: string;
  date: string;
  time: string | null;
  status: "future" | "past";
  circuit?: Circuit;
};

export type DriverOption = {
  driverId: number;
  number: string | null;
  constructorId: number;
  name: string;
  teamName: string;
  teamColor: string | null;
  label: string;
};

export type ConstructorOption = {
  constructorId: number;
  name: string;
  label: string;
};

export type LocalF1Data = {
  races: Race[];
  circuits: Circuit[];
  drivers: DriverOption[];
  constructors: ConstructorOption[];
  latestParticipants: ParticipantRequest[];
};

export type SavedPrediction = {
  id: string;
  created_at: string;
  source: "predicts" | "races";
  simulation_count: number;
  race: Pick<Race, "raceId" | "circuitId" | "name" | "date">;
  request: PredictionRequest;
  averaged_predictions: PredictionItem[];
  raw_predictions?: PredictionResponse[];
};
