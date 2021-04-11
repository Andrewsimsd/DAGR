using System;
using System.IO;
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
using System.Threading;
using DataUtil;
using DAGRCommon;

namespace DAGR
{
    public partial class DashboardForm : Form
    {
        public DAGRCommon.DAGRCommon dagrCommon;

        public DashboardForm()
        {
            InitializeComponent();
            InitalizeChartDefaults();

            dagrCommon = new DAGRCommon.DAGRCommon();            

            // Debugging relative path dataset
            // Navaigate back up from 'debug' and into datasets directory for relative pathing
            string datasetPath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.Parent.FullName,
                @"datasets\dataset_100.csv");

            tbImportFile.Text = datasetPath;            
        }

        private void InitalizeChartDefaults()
        {
            pnlNav.Height = btnImportData.Height;
            pnlNav.Top = btnImportData.Top;
            pnlNav.Left = btnImportData.Left;
            btnImportData.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR1;
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
        }

        private void btnDashboard_Click(object sender, EventArgs e)
        {
            pnlNav.Height = btnDashboard.Height;
            pnlNav.Top = btnDashboard.Top;
            pnlNav.Left = btnDashboard.Left;
            customTabControl1.SelectedTab = tabDashboard;
            btnDashboard.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR1;
            btnAnalysis.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR2;
            btnImportData.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR2;
        }

        private void btnAnalysis_Click(object sender, EventArgs e)
        {
            
            pnlNav.Height = btnAnalysis.Height;
            pnlNav.Top = btnAnalysis.Top;
            pnlNav.Left = btnAnalysis.Left;
            customTabControl1.SelectedTab = tabAnalysis;
            btnDashboard.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR2; 
            btnAnalysis.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR1;
            btnImportData.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR2;
        }

        private void btnImportData_Click(object sender, EventArgs e)
        {
            pnlNav.Height = btnImportData.Height;
            pnlNav.Top = btnImportData.Top;
            pnlNav.Left = btnImportData.Left;
            customTabControl1.SelectedTab = tabImportData;
            btnDashboard.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR2;
            btnAnalysis.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR2;
            btnImportData.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR1;

        }

        private void btnDashboard_Leave(object sender, EventArgs e)
        {
            btnDashboard.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR2;
        }

        private void btnAnalysis_Leave(object sender, EventArgs e)
        {
            btnAnalysis.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR2;
        }

        private void btnImportData_Leave(object sender, EventArgs e)
        {
            btnImportData.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR2;
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
            dagrCommon.flightDataList = DataUtil.DAGRReader.ReadFlightData(tbImportFile.Text);
            //Ideally this will actually check if it did load in the data correctly or not
            MessageBox.Show("Successfully loaded data");
            UpdateCharts(dagrCommon.flightDataList);
        }

        private void UpdateCharts(List<FlightData> flightDataList)
        {
            //Thread updateAltitudeChartThread = new Thread(() => UpdateAltitudeChart(flightData.altitude));
            //updateAltitudeChartThread.SetApartmentState(ApartmentState.STA);
            //updateAltitudeChartThread.Start();

            cartesianChartAltitude.Series = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Aircraft 1",
                    Values = new ChartValues<float>(flightDataList.Select(x => x.Altitude).ToArray()),
                    PointGeometry = null
                },
            };
            cartesianChartPitch.Series = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Aircraft 1",
                    Values = new ChartValues<float>(flightDataList.Select(x => x.Pitch).ToArray()),
                    PointGeometry = null
                },
            };

            cartesianChartRoll.Series = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Aircraft 1",
                    Values = new ChartValues<float>(flightDataList.Select(x => x.Roll).ToArray()),
                    PointGeometry = null
                },
            };

            cartesianChartYaw.Series = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Aircraft 1",
                    Values = new ChartValues<float>(flightDataList.Select(x => x.Yaw).ToArray()),
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
