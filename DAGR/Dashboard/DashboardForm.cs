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
using ScottPlot;

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
                @"datasets\dataset_1000000.csv");

            tbImportFile.Text = datasetPath;            
        }

        private void InitalizeChartDefaults()
        {
            pnlNav.Height = btnImportData.Height;
            pnlNav.Top = btnImportData.Top;
            pnlNav.Left = btnImportData.Left;
            btnImportData.BackColor = DAGRCommon.DAGRCommon.BACKGROUNDCOLOR1;
            customTabControl1.SelectedTab = tabImportData;
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
            //TODO Ideally this will actually check if it did load in the data correctly or not
            MessageBox.Show("Successfully loaded data");
            //On error of readflightdata, flightdatalist will be empty
            UpdateCharts(dagrCommon.flightDataList);
        }

        private void UpdateCharts(List<FlightData> flightDataList)
        {
            //Thread updateAltitudeChartThread = new Thread(() => UpdateAltitudeChart(flightData.altitude));
            //updateAltitudeChartThread.SetApartmentState(ApartmentState.STA);
            //updateAltitudeChartThread.Start();

            string attrName = "Altitude";
            // formsPlotAltVsTime

            double[] dataToPlot = flightDataList.Select(x => x.Altitude).ToArray();
            double[] altIndexes = Enumerable.Range(1, dataToPlot.Length).Select(x => (double)x).ToArray();
            formsPlotAltVsTime.plt.PlotScatter(altIndexes, dataToPlot, label: attrName);
            formsPlotAltVsTime.plt.Legend();
            formsPlotAltVsTime.plt.Title(attrName + " vs Time");
            formsPlotAltVsTime.plt.YLabel(attrName);
            formsPlotAltVsTime.plt.XLabel("Time");

            //new ScottPlot.FormsPlotViewer(plt).Show();
            formsPlotAltVsTime.Render();

            // formsPlotPitchVsTime
            attrName = "Pitch";
            dataToPlot = flightDataList.Select(x => x.Pitch).ToArray();            
            formsPlotPitchVsTime.plt.PlotScatter(altIndexes, dataToPlot, label: attrName);
            formsPlotPitchVsTime.plt.Legend();
            formsPlotPitchVsTime.plt.Title(attrName + " vs Time");
            formsPlotPitchVsTime.plt.YLabel(attrName);
            formsPlotPitchVsTime.plt.XLabel("Time");
            formsPlotPitchVsTime.Render();

            // formsPlotRollVsTime
            attrName = "Roll";
            dataToPlot = flightDataList.Select(x => x.Roll).ToArray();            
            formsPlotRollVsTime.plt.PlotScatter(altIndexes, dataToPlot, label: attrName);
            formsPlotRollVsTime.plt.Legend();
            formsPlotRollVsTime.plt.Title(attrName + " vs Time");
            formsPlotRollVsTime.plt.YLabel(attrName);
            formsPlotRollVsTime.plt.XLabel("Time");
            formsPlotRollVsTime.Render();

            // 
            attrName = "Yaw";
            dataToPlot = flightDataList.Select(x => x.Yaw).ToArray();            
            formsPlotYawVsTime.plt.PlotScatter(altIndexes, dataToPlot, label: attrName);
            formsPlotYawVsTime.plt.Legend();
            formsPlotYawVsTime.plt.Title(attrName + " vs Time");
            formsPlotYawVsTime.plt.YLabel(attrName);
            formsPlotYawVsTime.plt.XLabel("Time");
            formsPlotYawVsTime.Render();


        }
        //private void UpdateAltitudeChart(List<float> data)
        //{
        //
        //}

    }
    
}
