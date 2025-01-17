using Shared.Interfaces;
using System.Diagnostics;
using System.Text;
using Shared;

namespace BLL.Ai.Services
{
    public class PythonScriptService : IThirdPartyAiService, IPythonScriptService, IDisposable
    {
        private readonly string _scriptPath;
        private Process _process;
        private StreamWriter _writer;
        private StringBuilder _outputBuffer;
        private TaskCompletionSource<bool> _readinessTask;

        public PythonScriptService(string scriptPath)
        {
            _scriptPath = scriptPath ?? throw new ArgumentNullException(nameof(scriptPath));

            if (!File.Exists(_scriptPath))
            {
                throw new FileNotFoundException($"Python script not found at {_scriptPath}");
            }
        }

        #region IThirdPartyAiService

        public async Task<string> GetHabitToTrackSuggestion()
        {
            var result = await SendInputAsync(Constants.HABIT_TO_TRACK_PROMPT);

            return result;
        }

        #endregion

        #region IPythonScriptService

        public async Task StartAsync()
        {
            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{_scriptPath}\"",
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            _outputBuffer = new StringBuilder();
            _readinessTask = new TaskCompletionSource<bool>();

            _process = new Process { StartInfo = psi };

            // Attach output and error handlers
            _process.OutputDataReceived += (sender, args) =>
            {
                if (!string.IsNullOrWhiteSpace(args.Data))
                {
                    Debug.WriteLine($"[Python Output]: {args.Data}");
                    _outputBuffer.AppendLine(args.Data);

                    // Signal readiness if a specific line is detected
                    if (args.Data.Contains("Python model ready", StringComparison.OrdinalIgnoreCase))
                    {
                        _readinessTask.TrySetResult(true);
                    }
                }
            };

            _process.ErrorDataReceived += (sender, args) =>
            {
                if (!string.IsNullOrWhiteSpace(args.Data))
                {
                    Debug.WriteLine($"[Python Error]: {args.Data}");
                }
            };

            _process.Start();

            // Start reading output and error streams asynchronously
            _process.BeginOutputReadLine();
            _process.BeginErrorReadLine();

            _writer = _process.StandardInput;

            // Wait for readiness signal
            await _readinessTask.Task;
        }

        public async Task<string> SendInputAsync(string input)
        {
            if (_process == null || _process.HasExited)
            {
                throw new InvalidOperationException("Python process is not running.");
            }

            await _writer.WriteLineAsync(input);
            await _writer.FlushAsync();

            // Wait for the next output line asynchronously
            var output = await ReadNextOutputAsync();
            return output;
        }

        public void Stop()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (_process != null && !_process.HasExited)
            {
                _writer.WriteLine("exit");
                _writer.Flush();
                _process.WaitForExit();
            }

            _writer?.Dispose();
            _process?.Dispose();
        }

        #endregion

        #region Private Methods

        private async Task<string> ReadNextOutputAsync()
        {
            var tcs = new TaskCompletionSource<string>();
            DataReceivedEventHandler handler = null;

            handler = (sender, args) =>
            {
                if (!string.IsNullOrWhiteSpace(args.Data))
                {
                    tcs.TrySetResult(args.Data);
                }
            };

            _process.OutputDataReceived += handler;

            try
            {
                return await tcs.Task;
            }
            finally
            {
                _process.OutputDataReceived -= handler;
            }
        }

        #endregion
    }
}
