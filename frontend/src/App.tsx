import { useMemo, useState } from 'react'
import type { ChangeEvent } from 'react'
import './App.css'

type ModelName = 'ANN' | 'Elman' | 'TNet' | 'PLSR' | 'LSSVR'

type HistoryItem = {
  id: string
  model: ModelName
  moisture: number
  stage: string
  time: string
}

type SpectrumPoint = {
  wavelength: number
  value: number
}

const models: ModelName[] = ['ANN', 'Elman', 'TNet', 'PLSR', 'LSSVR']

const modelProfiles: Record<ModelName, { moisture: number; confidence: number; drift: string }> = {
  ANN: { moisture: 62.35, confidence: 96.8, drift: '输出稳定' },
  Elman: { moisture: 61.92, confidence: 94.6, drift: '序列收敛' },
  TNet: { moisture: 63.18, confidence: 95.2, drift: '特征清晰' },
  PLSR: { moisture: 60.74, confidence: 91.9, drift: '线性拟合' },
  LSSVR: { moisture: 62.81, confidence: 93.7, drift: '核函数正常' },
}

const initialHistory: HistoryItem[] = [
  { id: 'TW-260502-004', model: 'ANN', moisture: 62.35, stage: '中度萎凋', time: '2026-05-02 17:24' },
  { id: 'TW-260502-003', model: 'TNet', moisture: 67.48, stage: '轻度萎凋', time: '2026-05-02 16:48' },
  { id: 'TW-260502-002', model: 'PLSR', moisture: 71.06, stage: '鲜叶期', time: '2026-05-02 15:36' },
  { id: 'TW-260502-001', model: 'LSSVR', moisture: 55.27, stage: '重度萎凋', time: '2026-05-02 14:12' },
]

const formatDateTime = (date: Date) => {
  const pad = (value: number) => String(value).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}`
}

const stageFromMoisture = (moisture: number) => {
  if (moisture >= 70) return '鲜叶期'
  if (moisture >= 64) return '轻度萎凋'
  if (moisture >= 58) return '中度萎凋'
  return '重度萎凋'
}

const buildSpectrum = (model: ModelName, fileName: string): SpectrumPoint[] => {
  const modelOffset = models.indexOf(model) * 0.018
  const fileOffset = fileName.length ? (fileName.length % 9) * 0.006 : 0

  return Array.from({ length: 72 }, (_, index) => {
    const wavelength = 900 + index * 14
    const broadWaterBand = Math.exp(-Math.pow((wavelength - 1450) / 155, 2)) * 0.34
    const leafCelluloseBand = Math.exp(-Math.pow((wavelength - 1190) / 120, 2)) * 0.16
    const ripple = Math.sin(index * 0.42) * 0.024 + Math.cos(index * 0.19) * 0.018
    const value = 0.46 + broadWaterBand + leafCelluloseBand + ripple + modelOffset + fileOffset
    return { wavelength, value: Number(value.toFixed(3)) }
  })
}

const makePath = (points: SpectrumPoint[], width: number, height: number, padding: number) => {
  const minX = Math.min(...points.map((point) => point.wavelength))
  const maxX = Math.max(...points.map((point) => point.wavelength))
  const minY = Math.min(...points.map((point) => point.value))
  const maxY = Math.max(...points.map((point) => point.value))
  const xScale = (value: number) => padding + ((value - minX) / (maxX - minX)) * (width - padding * 2)
  const yScale = (value: number) =>
    height - padding - ((value - minY) / (maxY - minY)) * (height - padding * 2)

  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xScale(point.wavelength).toFixed(2)} ${yScale(point.value).toFixed(2)}`)
    .join(' ')
}

const mapChartPoint = (points: SpectrumPoint[], point: SpectrumPoint, width: number, height: number, padding: number) => {
  const minX = Math.min(...points.map((item) => item.wavelength))
  const maxX = Math.max(...points.map((item) => item.wavelength))
  const minY = Math.min(...points.map((item) => item.value))
  const maxY = Math.max(...points.map((item) => item.value))

  return {
    x: padding + ((point.wavelength - minX) / (maxX - minX)) * (width - padding * 2),
    y: height - padding - ((point.value - minY) / (maxY - minY)) * (height - padding * 2),
  }
}

function App() {
  const [selectedModel, setSelectedModel] = useState<ModelName>('ANN')
  const [fileName, setFileName] = useState('sample_spectrum_tea_004.csv')
  const [sampleId, setSampleId] = useState('TW-260502-004')
  const [detectionTime, setDetectionTime] = useState('2026-05-02 17:24')
  const [history, setHistory] = useState<HistoryItem[]>(initialHistory)

  const prediction = modelProfiles[selectedModel]
  const stage = stageFromMoisture(prediction.moisture)
  const spectrumData = useMemo(() => buildSpectrum(selectedModel, fileName), [selectedModel, fileName])
  const chartPath = useMemo(() => makePath(spectrumData, 760, 300, 36), [spectrumData])
  const filledPath = `${chartPath} L 724 264 L 36 264 Z`
  const latestValue = spectrumData[spectrumData.length - 1]
  const latestPoint = useMemo(() => mapChartPoint(spectrumData, latestValue, 760, 300, 36), [spectrumData, latestValue])
  const spectrumSummary = useMemo(() => {
    const values = spectrumData.map((point) => point.value)
    const peak = spectrumData.reduce((max, point) => (point.value > max.value ? point : max), spectrumData[0])
    const minValue = Math.min(...values)
    const maxValue = Math.max(...values)
    const meanValue = values.reduce((sum, value) => sum + value, 0) / values.length

    return {
      peak,
      minValue,
      maxValue,
      meanValue,
      range: maxValue - minValue,
    }
  }, [spectrumData])

  const handleUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const now = new Date()
    const nextSampleId = `TW-${String(now.getFullYear()).slice(2)}${String(now.getMonth() + 1).padStart(2, '0')}${String(
      now.getDate(),
    ).padStart(2, '0')}-${String(history.length + 5).padStart(3, '0')}`
    const nextTime = formatDateTime(now)
    const nextStage = stageFromMoisture(prediction.moisture)

    setFileName(file.name)
    setSampleId(nextSampleId)
    setDetectionTime(nextTime)
    setHistory((items) => [
      { id: nextSampleId, model: selectedModel, moisture: prediction.moisture, stage: nextStage, time: nextTime },
      ...items.slice(0, 5),
    ])
  }

  return (
    <main className="dashboard-shell">
      <header className="title-band">
        <div className="title-copy">
          <span className="eyebrow">NIR Tea Lab / Moisture Analytics</span>
          <h1>基于近红外光谱的茶叶萎凋过程水分检测系统</h1>
          <p>Tea Withering Moisture Detection System</p>
        </div>
        <div className="lab-signal" aria-label="当前系统状态">
          <span className="signal-dot" />
          <div>
            <strong>实验台在线</strong>
            <small>API 预留接入 · 模拟预测</small>
          </div>
        </div>
      </header>

      <section className="summary-strip" aria-label="当前检测摘要">
        <div>
          <span>样本编号</span>
          <strong>{sampleId}</strong>
        </div>
        <div>
          <span>检测时间</span>
          <strong>{detectionTime}</strong>
        </div>
        <div>
          <span>当前模型</span>
          <strong>{selectedModel}</strong>
        </div>
        <div>
          <span>光谱点位</span>
          <strong>{spectrumData.length} 点</strong>
        </div>
      </section>

      <section className="dashboard-grid">
        <div className="dashboard-column left-column">
          <section className="panel upload-panel">
            <div className="panel-heading">
              <span>01</span>
              <div>
                <h2>光谱数据上传</h2>
                <p>txt / csv 近红外光谱文件</p>
              </div>
            </div>

            <label className="upload-zone">
              <input type="file" accept=".txt,.csv,text/plain,text/csv" onChange={handleUpload} />
              <span className="upload-mark">NIR</span>
              <strong>选择或拖入光谱数据</strong>
              <small>前端展示检测流程，后续由后端 API 调用 Python 模型</small>
            </label>

            <dl className="file-meta">
              <div>
                <dt>文件名</dt>
                <dd>{fileName}</dd>
              </div>
              <div>
                <dt>样本编号</dt>
                <dd>{sampleId}</dd>
              </div>
              <div>
                <dt>检测时间</dt>
                <dd>{detectionTime}</dd>
              </div>
            </dl>
          </section>

          <section className="panel model-panel">
            <div className="panel-heading compact">
              <span>02</span>
              <div>
                <h2>模型选择</h2>
                <p>切换模拟输出</p>
              </div>
            </div>
            <div className="model-grid">
              {models.map((model) => (
                <button
                  className={model === selectedModel ? 'model-button active' : 'model-button'}
                  key={model}
                  onClick={() => setSelectedModel(model)}
                  type="button"
                >
                  {model}
                </button>
              ))}
            </div>
          </section>

          <section className="panel params-panel">
            <div className="panel-heading compact">
              <span>03</span>
              <div>
                <h2>实验参数</h2>
                <p>茶叶萎凋检测条件</p>
              </div>
            </div>
            <div className="parameter-list">
              <div>
                <span>环境温度</span>
                <strong>24.6 C</strong>
              </div>
              <div>
                <span>相对湿度</span>
                <strong>68%</strong>
              </div>
              <div>
                <span>积分时间</span>
                <strong>120 ms</strong>
              </div>
              <div>
                <span>扫描次数</span>
                <strong>32</strong>
              </div>
            </div>
          </section>
        </div>

        <div className="dashboard-column center-column">
          <section className="panel result-panel">
            <div className="panel-heading">
              <span>04</span>
              <div>
                <h2>水分预测结果!</h2>
                <p>当前模型输出与萎凋状态</p>
              </div>
            </div>

            <div className="result-content">
              <div className="moisture-readout">
                <span>预测水分含量</span>
                <strong>{prediction.moisture.toFixed(2)}%</strong>
              </div>

              <div className="result-metrics">
                <div>
                  <span>萎凋状态</span>
                  <strong>{stage}</strong>
                </div>
                <div>
                  <span>模型状态</span>
                  <strong>{prediction.drift}</strong>
                </div>
              </div>
            </div>

            <div className="confidence-block">
              <div>
                <span>置信度</span>
                <strong>{prediction.confidence.toFixed(1)}%</strong>
              </div>
              <div className="confidence-track">
                <span style={{ width: `${prediction.confidence}%` }} />
              </div>
            </div>
          </section>

          <section className="panel history-panel">
            <div className="panel-heading compact">
              <span>05</span>
              <div>
                <h2>预测历史记录</h2>
                <p>最近样本的模型输出与萎凋判定</p>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>样本编号</th>
                    <th>模型</th>
                    <th>预测水分</th>
                    <th>萎凋状态</th>
                    <th>检测时间</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item) => (
                    <tr key={`${item.id}-${item.time}`}>
                      <td>{item.id}</td>
                      <td>{item.model}</td>
                      <td>{item.moisture.toFixed(2)}%</td>
                      <td>{item.stage}</td>
                      <td>{item.time}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>

        <div className="dashboard-column right-column">
          <section className="panel chart-panel">
            <div className="panel-heading">
              <span>06</span>
              <div>
                <h2>近红外光谱曲线</h2>
                <p>横轴：波长 / 特征点，纵轴：吸光度 / 反射率</p>
              </div>
            </div>

            <div className="chart-wrap">
              <svg viewBox="0 0 760 300" role="img" aria-label="模拟近红外光谱曲线">
                <defs>
                  <linearGradient id="spectrumFill" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#4a8f57" stopOpacity="0.32" />
                    <stop offset="100%" stopColor="#4a8f57" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {[0, 1, 2, 3].map((line) => (
                  <line key={line} x1="36" x2="724" y1={52 + line * 53} y2={52 + line * 53} className="grid-line" />
                ))}
                {[0, 1, 2, 3, 4].map((line) => (
                  <line
                    key={line}
                    x1={36 + line * 172}
                    x2={36 + line * 172}
                    y1="36"
                    y2="264"
                    className="grid-line soft"
                  />
                ))}
                <path d={filledPath} className="spectrum-fill" />
                <path d={chartPath} className="spectrum-line" />
                <circle cx={latestPoint.x} cy={latestPoint.y} r="4.5" className="last-point" />
                <text x="36" y="288" className="axis-label">
                  900 nm
                </text>
                <text x="656" y="288" className="axis-label">
                  1894 nm
                </text>
                <text x="18" y="44" className="axis-label rotate">
                  A/R
                </text>
              </svg>
            </div>

            <div className="chart-stats">
              <span>末端波长 {latestValue.wavelength} nm</span>
              <span>响应值 {latestValue.value.toFixed(3)}</span>
              <span>水吸收峰 1450 nm</span>
            </div>
          </section>

          <section className="panel feature-panel">
            <div className="panel-heading compact">
              <span>07</span>
              <div>
                <h2>光谱特征摘要</h2>
                <p>模拟曲线关键响应值</p>
              </div>
            </div>
            <div className="feature-list">
              <div>
                <span>主吸收峰</span>
                <strong>
                  {spectrumSummary.peak.wavelength} nm / {spectrumSummary.peak.value.toFixed(3)}
                </strong>
              </div>
              <div>
                <span>响应均值</span>
                <strong>{spectrumSummary.meanValue.toFixed(3)}</strong>
              </div>
              <div>
                <span>响应范围</span>
                <strong>
                  {spectrumSummary.minValue.toFixed(3)} - {spectrumSummary.maxValue.toFixed(3)}
                </strong>
              </div>
              <div>
                <span>峰谷差</span>
                <strong>{spectrumSummary.range.toFixed(3)}</strong>
              </div>
            </div>
          </section>
        </div>
      </section>
    </main>
  )
}

export default App
