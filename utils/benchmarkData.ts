export interface BenchmarkResult {
  release_date: string;
  agents: {
    [key: string]: {
      p50_horizon_length?: {
        estimate: number;
        ci_low: number;
        ci_high: number;
      };
      p80_horizon_length?: {
        estimate: number;
        ci_low: number;
        ci_high: number;
      };
      average_score?: {
        estimate: number;
      };
      usage?: {
        working_time: number;
        usd: number;
      };
    };
  };
  always_show?: boolean;
}

export interface BenchmarkData {
  benchmark_name: string;
  long_tasks_version: string;
  swaa_version: string;
  results: {
    [key: string]: BenchmarkResult;
  };
}

export const benchmarkData: BenchmarkData = {
  benchmark_name: "METR-Horizon-v1.1",
  long_tasks_version: "799cc9c4b4483a93fc3445623a49ea1bd74fdeb2",
  swaa_version: "f6cc84052e2a79dd540766c8a1b15ff399696371",
  results: {
    claude_3_5_sonnet_20240620_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 11.395377,
            ci_low: 5.489734,
            ci_high: 22.384214
          },
          p80_horizon_length: {
            estimate: 1.671757,
            ci_low: 0.57927,
            ci_high: 4.573519
          },
          average_score: {
            estimate: 0.401502
          }
        }
      },
      release_date: "2024-06-20",
    },
    claude_3_5_sonnet_20241022_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 20.522872,
            ci_low: 10.144026,
            ci_high: 40.82028
          },
          p80_horizon_length: {
            estimate: 2.595677,
            ci_low: 0.89379,
            ci_high: 7.363864
          },
          average_score: {
            estimate: 0.452356
          }
        }
      },
      release_date: "2024-10-22",
    },
    claude_3_7_sonnet_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 60.388937,
            ci_low: 33.006168,
            ci_high: 104.226017
          },
          p80_horizon_length: {
            estimate: 12.09179,
            ci_low: 4.575608,
            ci_high: 28.889879
          },
          average_score: {
            estimate: 0.558217
          }
        }
      },
      release_date: "2025-02-24",
    },
    claude_3_opus_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 3.952262,
            ci_low: 1.706313,
            ci_high: 8.76484
          },
          p80_horizon_length: {
            estimate: 0.638973,
            ci_low: 0.190491,
            ci_high: 2.118276
          },
          average_score: {
            estimate: 0.294753
          }
        }
      },
      release_date: "2024-03-04",
    },
    claude_4_1_opus_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 100.472004,
            ci_low: 59.272391,
            ci_high: 159.451918
          },
          p80_horizon_length: {
            estimate: 23.455761,
            ci_low: 10.538456,
            ci_high: 47.785951
          },
          average_score: {
            estimate: 0.615713
          }
        }
      },
      release_date: "2025-08-05",
    },
    claude_4_opus_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 100.366123,
            ci_low: 59.994073,
            ci_high: 163.454074
          },
          p80_horizon_length: {
            estimate: 20.429752,
            ci_low: 8.483125,
            ci_high: 41.937268
          },
          average_score: {
            estimate: 0.615452
          }
        }
      },
      release_date: "2025-05-22",
    },
    claude_opus_4_5_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 292.994594,
            ci_low: 161.717714,
            ci_high: 623.704698
          },
          p80_horizon_length: {
            estimate: 49.430584,
            ci_low: 20.660485,
            ci_high: 104.641318
          },
          average_score: {
            estimate: 0.73007
          }
        }
      },
      release_date: "2025-11-24",
    },
    claude_opus_4_6_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 718.80683,
            ci_low: 316.685725,
            ci_high: 3633.786163
          },
          p80_horizon_length: {
            estimate: 69.874587,
            ci_low: 27.026521,
            ci_high: 170.437873
          },
          average_score: {
            estimate: 0.788644
          }
        }
      },
      release_date: "2026-02-05",
    },
    davinci_002: {
      agents: {
        "modular-public": {
          p50_horizon_length: {
            estimate: 0.144057,
            ci_low: 0.093319,
            ci_high: 0.221209
          },
          p80_horizon_length: {
            estimate: 0.056238,
            ci_low: 0.034813,
            ci_high: 0.105813
          },
          average_score: {
            estimate: 0.161869
          }
        }
      },
      release_date: "2020-05-28",
    },
    gpt_5_3_codex: {
      agents: {
        "mtb/start_metr_task,triframe_inspect/triframe_agent": {
          p50_horizon_length: {
            estimate: 349.530732,
            ci_low: 194.913134,
            ci_high: 816.351994
          },
          p80_horizon_length: {
            estimate: 54.739407,
            ci_low: 22.361504,
            ci_high: 122.058963
          },
          average_score: {
            estimate: 0.745439
          }
        }
      },
      release_date: "2026-02-05",
    },
    gemini_3_pro: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 224.325884,
            ci_low: 139.565157,
            ci_high: 379.235434
          },
          p80_horizon_length: {
            estimate: 54.142849,
            ci_low: 25.650451,
            ci_high: 103.295501
          },
          average_score: {
            estimate: 0.709822
          }
        }
      },
      release_date: "2025-11-18",
    },
    gemini_3_1_pro: {
      agents: {
        "mtb/start_metr_task,triframe_inspect/triframe_agent": {
          p50_horizon_length: {
            estimate: 384.147435,
            ci_low: 233.50073,
            ci_high: 694.750898
          },
          p80_horizon_length: {
            estimate: 89.801503,
            ci_low: 52.025934,
            ci_high: 158.618017
          },
          average_score: {
            estimate: 0.77035
          }
        }
      },
      release_date: "2026-02-19",
    },
    gpt2: {
      agents: {
        "None": {
          p50_horizon_length: {
            estimate: 0.053778,
            ci_low: 0.009994,
            ci_high: 0.141825
          },
          p80_horizon_length: {
            estimate: 0.0128,
            ci_low: 0.001589,
            ci_high: 0.068612
          },
          average_score: {
            estimate: 0.101046
          }
        }
      },
      release_date: "2019-02-14",
    },
    gpt_3_5_turbo_instruct: {
      agents: {
        "modular-public": {
          p50_horizon_length: {
            estimate: 0.599247,
            ci_low: 0.255148,
            ci_high: 1.115905
          },
          p80_horizon_length: {
            estimate: 0.25538,
            ci_low: 0.141649,
            ci_high: 0.492933
          },
          average_score: {
            estimate: 0.214611
          }
        }
      },
      release_date: "2022-03-15",
    },
    gpt_4: {
      agents: {
        "modular-public": {
          p50_horizon_length: {
            estimate: 3.987428,
            ci_low: 1.93292,
            ci_high: 7.995283
          },
          p80_horizon_length: {
            estimate: 0.889561,
            ci_low: 0.342984,
            ci_high: 2.523746
          },
          average_score: {
            estimate: 0.293044
          }
        }
      },
      release_date: "2023-03-14",
    },
    gpt_4_1106_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 4.044959,
            ci_low: 1.866859,
            ci_high: 8.443226
          },
          p80_horizon_length: {
            estimate: 0.783032,
            ci_low: 0.276599,
            ci_high: 2.358414
          },
          average_score: {
            estimate: 0.28905
          }
        }
      },
      release_date: "2023-11-06",
    },
    gpt_4_turbo_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 3.732787,
            ci_low: 1.980046,
            ci_high: 6.736613
          },
          p80_horizon_length: {
            estimate: 0.927933,
            ci_low: 0.428277,
            ci_high: 2.196806
          },
          average_score: {
            estimate: 0.271786
          }
        }
      },
      release_date: "2024-04-09",
    },
    gpt_4o_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 6.991195,
            ci_low: 4.001482,
            ci_high: 12.905741
          },
          p80_horizon_length: {
            estimate: 1.267009,
            ci_low: 0.562997,
            ci_high: 3.01517
          },
          average_score: {
            estimate: 0.338424
          }
        }
      },
      release_date: "2024-05-13",
    },
    gpt_5_1_codex_max_inspect: {
      agents: {
        "mtb/start_metr_task,triframe_inspect/triframe_agent": {
          p50_horizon_length: {
            estimate: 223.714694,
            ci_low: 134.31077,
            ci_high: 396.206723
          },
          p80_horizon_length: {
            estimate: 50.632499,
            ci_low: 27.30204,
            ci_high: 88.365435
          },
          average_score: {
            estimate: 0.708301
          }
        }
      },
      release_date: "2025-11-19",
    },
    gpt_5_2: {
      agents: {
        "mtb/start_metr_task,triframe_inspect/triframe_agent": {
          p50_horizon_length: {
            estimate: 352.249302,
            ci_low: 198.067494,
            ci_high: 815.177445
          },
          p80_horizon_length: {
            estimate: 66.002649,
            ci_low: 31.665111,
            ci_high: 131.730719
          },
          average_score: {
            estimate: 0.752875
          }
        }
      },
      release_date: "2025-12-11",
    },
    gpt_5_2025_08_07_inspect: {
      agents: {
        "mtb/start_metr_task,triframe_inspect/triframe_agent": {
          p50_horizon_length: {
            estimate: 203.012577,
            ci_low: 112.641357,
            ci_high: 405.551565
          },
          p80_horizon_length: {
            estimate: 38.312431,
            ci_low: 19.184874,
            ci_high: 70.093125
          },
          average_score: {
            estimate: 0.693697
          }
        }
      },
      release_date: "2025-08-07",
    },
    gpt_5_4: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 341.735276,
            ci_low: 186.581591,
            ci_high: 768.779526
          },
          p80_horizon_length: {
            estimate: 53.877851,
            ci_low: 23.957027,
            ci_high: 108.679232
          },
          average_score: {
            estimate: 0.743443
          }
        }
      },
      release_date: "2026-03-05",
    },
    o1_inspect: {
      agents: {
        "mtb/start_metr_task,triframe_inspect/triframe_agent": {
          p50_horizon_length: {
            estimate: 38.831588,
            ci_low: 21.164512,
            ci_high: 64.95249
          },
          p80_horizon_length: {
            estimate: 7.090121,
            ci_low: 3.036033,
            ci_high: 16.640636
          },
          average_score: {
            estimate: 0.510512
          }
        }
      },
      release_date: "2024-12-05",
    },
    o1_preview: {
      agents: {
        "duet": {
          p50_horizon_length: {
            estimate: 20.326586,
            ci_low: 11.716193,
            ci_high: 33.379877
          },
          p80_horizon_length: {
            estimate: 4.420545,
            ci_low: 2.012646,
            ci_high: 8.912444
          },
          average_score: {
            estimate: 0.451441
          }
        }
      },
      release_date: "2024-09-12",
    },
    o3_inspect: {
      agents: {
        "mtb/start_metr_task,triframe_inspect/triframe_agent": {
          p50_horizon_length: {
            estimate: 119.732634,
            ci_low: 74.615398,
            ci_high: 190.943818
          },
          p80_horizon_length: {
            estimate: 29.981603,
            ci_low: 15.127348,
            ci_high: 57.823511
          },
          average_score: {
            estimate: 0.636235
          }
        }
      },
      release_date: "2025-04-16",
    },
    claude_mythos_preview_early_inspect: {
      agents: {
        "mtb/start_metr_task,metr_agents/react": {
          p50_horizon_length: {
            estimate: 1044.780145,
            ci_low: 508.876789,
            ci_high: 3304.261235
          },
          p80_horizon_length: {
            estimate: 185.911829,
            ci_low: 97.30292,
            ci_high: 398.514614
          },
          average_score: {
            estimate: 0.85205
          }
        }
      },
      release_date: "2026-04-07",
    }
  }
};