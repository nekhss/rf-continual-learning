export const tasks = [
  { id: 1, label: "T1", snr: "+12 to +18 dB", values: [12, 14, 16, 18] },
  { id: 2, label: "T2", snr: "+4 to +10 dB", values: [4, 6, 8, 10] },
  { id: 3, label: "T3", snr: "-4 to +2 dB", values: [-4, -2, 0, 2] },
  { id: 4, label: "T4", snr: "-12 to -6 dB", values: [-12, -10, -8, -6] },
  { id: 5, label: "T5", snr: "-20 to -14 dB", values: [-20, -18, -16, -14] }
];

export const methods = {
  "Naive Sequential": {
    color: "cyan",
    matrix: [
      [0.8205],
      [0.8127, 0.8665],
      [0.7542, 0.7955, 0.7705],
      [0.3139, 0.3211, 0.3482, 0.3591],
      [0.1408, 0.1386, 0.1394, 0.1988, 0.1453]
    ]
  },
  "Random Replay": {
    color: "violet",
    matrix: [
      [0.8402],
      [0.8370, 0.8430],
      [0.8242, 0.8242, 0.7692],
      [0.7623, 0.7450, 0.5689, 0.3600],
      [0.6329, 0.6282, 0.4879, 0.2792, 0.1418]
    ],
    aa: 0.4340,
    af: 0.1961
  },
  "SNR-Aware Replay": {
    color: "lime",
    matrix: [
      [0.8402],
      [0.8508, 0.8573],
      [0.8188, 0.8262, 0.7789],
      [0.7750, 0.7658, 0.5476, 0.3592],
      [0.7492, 0.7495, 0.5876, 0.2339, 0.1444]
    ],
    aa: 0.49294,
    af: 0.12883
  },
  "Joint Training": {
    color: "amber",
    matrix: [],
    pending: true
  }
};

export const modulationClasses = [
  "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
  "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"
];

export function finalAcc(method) {
  const m = methods[method];
  return m.matrix?.length === 5 ? m.matrix[4] : [];
}
