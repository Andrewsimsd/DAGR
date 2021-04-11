using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using DataUtil;
using System.Threading;

namespace Dashboard
{
    public partial class DAGR : Form
    {
        public DAGR()
        {
            InitializeComponent();
            pnlNav.Height = btnImportData.Height;
            pnlNav.Top = btnImportData.Top;
            pnlNav.Left = btnImportData.Left;
            btnImportData.BackColor = Color.FromArgb(28, 28, 28);
            customTabControl1.SelectedTab = tabImportData;
            cartesianChartYaw.Zoom = ZoomingOptions.X;
            cartesianChartYaw.Pan = PanningOptions.X;
            cartesianChartRoll.Zoom = ZoomingOptions.X;
            cartesianChartRoll.Pan = PanningOptions.X;
            cartesianChartPitch.Zoom = ZoomingOptions.X;
            cartesianChartPitch.Pan = PanningOptions.X;
            cartesianChartAltitude.Zoom = ZoomingOptions.X;
            cartesianChartAltitude.Pan = PanningOptions.X;

            cartesianChartYaw.AxisY.Add(new Axis
            {
                Title = "Yaw (°)",
            });
            cartesianChartYaw.AxisX.Add(new Axis
            {
                Title = "Time (HH:MM:SS)",
            });
            cartesianChartRoll.AxisY.Add(new Axis
            {
                Title = "Roll (°)",
            });
            cartesianChartRoll.AxisX.Add(new Axis
            {
                Title = "Time (HH:MM:SS)",
            });
            cartesianChartAltitude.AxisY.Add(new Axis
            {
                Title = "Altitude (Feet)",
            });
            cartesianChartAltitude.AxisX.Add(new Axis
            {
                Title = "Time (HH:MM:SS)",
            });
            cartesianChartPitch.AxisY.Add(new Axis
            {
                Title = "Pitch (°)",
            });
            cartesianChartPitch.AxisX.Add(new Axis
            {
                Title = "Time (HH:MM:SS)",
            });
            //hard code path for ease  of debug
            tbImportFile.Text = @"C:\Users\andre\Documents\Python Scripts\Data Analysis Intro\dataset_100.csv";
        }

        private void btnDashboard_Click(object sender, EventArgs e)
        {
            pnlNav.Height = btnDashboard.Height;
            pnlNav.Top = btnDashboard.Top;
            pnlNav.Left = btnDashboard.Left;
            customTabControl1.SelectedTab = tabDashboard;
            btnDashboard.BackColor = Color.FromArgb(28, 28, 28);
            btnAnalysis.BackColor = Color.FromArgb(45, 45, 48);
            btnImportData.BackColor = Color.FromArgb(45, 45, 48);
        }

        private void btnAnalysis_Click(object sender, EventArgs e)
        {
            
            pnlNav.Height = btnAnalysis.Height;
            pnlNav.Top = btnAnalysis.Top;
            pnlNav.Left = btnAnalysis.Left;
            customTabControl1.SelectedTab = tabAnalysis;
            btnDashboard.BackColor = Color.FromArgb(45, 45, 48);
            btnAnalysis.BackColor = Color.FromArgb(28, 28, 28);
            btnImportData.BackColor = Color.FromArgb(45, 45, 48);
        }

        private void btnImportData_Click(object sender, EventArgs e)
        {
            pnlNav.Height = btnImportData.Height;
            pnlNav.Top = btnImportData.Top;
            pnlNav.Left = btnImportData.Left;
            customTabControl1.SelectedTab = tabImportData;
            btnDashboard.BackColor = Color.FromArgb(45, 45, 48);
            btnAnalysis.BackColor = Color.FromArgb(45, 45, 48);
            btnImportData.BackColor = Color.FromArgb(28, 28, 28);

        }

        private void btnDashboard_Leave(object sender, EventArgs e)
        {
            btnDashboard.BackColor = Color.FromArgb(45, 45, 48);
        }

        private void btnAnalysis_Leave(object sender, EventArgs e)
        {
            btnAnalysis.BackColor = Color.FromArgb(45, 45, 48);
        }

        private void btnImportData_Leave(object sender, EventArgs e)
        {
            btnImportData.BackColor = Color.FromArgb(45, 45, 48);
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void btnBrowse_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "CSV File |*.csv";
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                tbImportFile.Text = ofd.FileName;
            }
        }

        private void btnLoad_Click(object sender, EventArgs e)
        {
            FlightData flightData = new FlightData();
            flightData.ReadCsv(tbImportFile.Text);
            UpdateCharts(flightData);
        }

        private void UpdateCharts(FlightData flightData)
        {
            //Thread updateAltitudeChartThread = new Thread(()=>UpdateAltitudeChart(flightData.altitude));
            //updateAltitudeChartThread.SetApartmentState(ApartmentState.STA);
            //updateAltitudeChartThread.Start();

            cartesianChartAltitude.Series = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Aircraft 1",
                    Values = new ChartValues<float>(flightData.altitude),
                    PointGeometry = null
                },
            };
            cartesianChartPitch.Series = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Aircraft 1",
                    Values = new ChartValues<float>(flightData.pitch),
                    PointGeometry = null
                },
            };
            
            cartesianChartRoll.Series = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Aircraft 1",
                    Values = new ChartValues<float>(flightData.roll),
                    PointGeometry = null
                },
            };
            
            cartesianChartYaw.Series = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Aircraft 1",
                    Values = new ChartValues<float>(flightData.yaw),
                    PointGeometry = null
                },
            };
            
        }
        //private void UpdateAltitudeChart(List<float> data)
        //{
        //
        //}

    }
    
}
