namespace DAGR
{
    partial class DashboardForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.panel1 = new System.Windows.Forms.Panel();
            this.pnlNav = new System.Windows.Forms.Panel();
            this.btnSettings = new System.Windows.Forms.Button();
            this.btnAnalysis = new System.Windows.Forms.Button();
            this.btnDashboard = new System.Windows.Forms.Button();
            this.btnImportData = new System.Windows.Forms.Button();
            this.panel2 = new System.Windows.Forms.Panel();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.customTabControl1 = new global::DAGR.CustomTabControl();
            this.tabDashboard = new System.Windows.Forms.TabPage();
            this.tabAnalysis = new System.Windows.Forms.TabPage();
            this.tableLayoutPanelMaster = new System.Windows.Forms.TableLayoutPanel();
            this.tableLayoutPanelYaw = new System.Windows.Forms.TableLayoutPanel();
            this.labelYaw = new System.Windows.Forms.Label();
            this.cartesianChartYaw = new LiveCharts.WinForms.CartesianChart();
            this.tableLayoutPanelRoll = new System.Windows.Forms.TableLayoutPanel();
            this.labelRoll = new System.Windows.Forms.Label();
            this.cartesianChartRoll = new LiveCharts.WinForms.CartesianChart();
            this.tableLayoutPanelPitch = new System.Windows.Forms.TableLayoutPanel();
            this.labelPitch = new System.Windows.Forms.Label();
            this.cartesianChartPitch = new LiveCharts.WinForms.CartesianChart();
            this.tableLayoutPanelAltitude = new System.Windows.Forms.TableLayoutPanel();
            this.labelAltitude = new System.Windows.Forms.Label();
            this.cartesianChartAltitude = new LiveCharts.WinForms.CartesianChart();
            this.tabImportData = new System.Windows.Forms.TabPage();
            this.btnLoad = new System.Windows.Forms.Button();
            this.tbImportFile = new System.Windows.Forms.TextBox();
            this.btnBrowse = new System.Windows.Forms.Button();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.customTabControl1.SuspendLayout();
            this.tabAnalysis.SuspendLayout();
            this.tableLayoutPanelMaster.SuspendLayout();
            this.tableLayoutPanelYaw.SuspendLayout();
            this.tableLayoutPanelRoll.SuspendLayout();
            this.tableLayoutPanelPitch.SuspendLayout();
            this.tableLayoutPanelAltitude.SuspendLayout();
            this.tabImportData.SuspendLayout();
            this.SuspendLayout();
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(45)))), ((int)(((byte)(45)))), ((int)(((byte)(48)))));
            this.panel1.Controls.Add(this.pnlNav);
            this.panel1.Controls.Add(this.btnSettings);
            this.panel1.Controls.Add(this.btnAnalysis);
            this.panel1.Controls.Add(this.btnDashboard);
            this.panel1.Controls.Add(this.btnImportData);
            this.panel1.Controls.Add(this.panel2);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Left;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(230, 795);
            this.panel1.TabIndex = 0;
            // 
            // pnlNav
            // 
            this.pnlNav.BackColor = System.Drawing.Color.White;
            this.pnlNav.Location = new System.Drawing.Point(0, 185);
            this.pnlNav.Name = "pnlNav";
            this.pnlNav.Size = new System.Drawing.Size(5, 100);
            this.pnlNav.TabIndex = 6;
            // 
            // btnSettings
            // 
            this.btnSettings.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.btnSettings.FlatAppearance.BorderSize = 0;
            this.btnSettings.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnSettings.Font = new System.Drawing.Font("Nirmala UI", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnSettings.ForeColor = System.Drawing.Color.White;
            this.btnSettings.Image = global::DAGR.Properties.Resources.baseline_settings_white_36dp;
            this.btnSettings.Location = new System.Drawing.Point(0, 735);
            this.btnSettings.Name = "btnSettings";
            this.btnSettings.Size = new System.Drawing.Size(230, 60);
            this.btnSettings.TabIndex = 5;
            this.btnSettings.Text = "Settings";
            this.btnSettings.TextImageRelation = System.Windows.Forms.TextImageRelation.TextBeforeImage;
            this.btnSettings.UseVisualStyleBackColor = true;
            // 
            // btnAnalysis
            // 
            this.btnAnalysis.Dock = System.Windows.Forms.DockStyle.Top;
            this.btnAnalysis.FlatAppearance.BorderSize = 0;
            this.btnAnalysis.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnAnalysis.Font = new System.Drawing.Font("Nirmala UI", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnAnalysis.ForeColor = System.Drawing.Color.White;
            this.btnAnalysis.Image = global::DAGR.Properties.Resources.baseline_query_stats_white_36dp;
            this.btnAnalysis.Location = new System.Drawing.Point(0, 285);
            this.btnAnalysis.Name = "btnAnalysis";
            this.btnAnalysis.Size = new System.Drawing.Size(230, 60);
            this.btnAnalysis.TabIndex = 3;
            this.btnAnalysis.Text = "Analysis";
            this.btnAnalysis.TextImageRelation = System.Windows.Forms.TextImageRelation.TextBeforeImage;
            this.btnAnalysis.UseVisualStyleBackColor = true;
            this.btnAnalysis.Click += new System.EventHandler(this.btnAnalysis_Click);
            this.btnAnalysis.Leave += new System.EventHandler(this.btnAnalysis_Leave);
            // 
            // btnDashboard
            // 
            this.btnDashboard.Dock = System.Windows.Forms.DockStyle.Top;
            this.btnDashboard.FlatAppearance.BorderSize = 0;
            this.btnDashboard.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnDashboard.Font = new System.Drawing.Font("Nirmala UI", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnDashboard.ForeColor = System.Drawing.Color.White;
            this.btnDashboard.Image = global::DAGR.Properties.Resources.baseline_home_white_36dp;
            this.btnDashboard.Location = new System.Drawing.Point(0, 225);
            this.btnDashboard.Name = "btnDashboard";
            this.btnDashboard.Size = new System.Drawing.Size(230, 60);
            this.btnDashboard.TabIndex = 2;
            this.btnDashboard.Text = "Dashboard";
            this.btnDashboard.TextImageRelation = System.Windows.Forms.TextImageRelation.TextBeforeImage;
            this.btnDashboard.UseVisualStyleBackColor = true;
            this.btnDashboard.Click += new System.EventHandler(this.btnDashboard_Click);
            this.btnDashboard.Leave += new System.EventHandler(this.btnDashboard_Leave);
            // 
            // btnImportData
            // 
            this.btnImportData.Dock = System.Windows.Forms.DockStyle.Top;
            this.btnImportData.FlatAppearance.BorderSize = 0;
            this.btnImportData.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnImportData.Font = new System.Drawing.Font("Nirmala UI", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnImportData.ForeColor = System.Drawing.Color.White;
            this.btnImportData.Image = global::DAGR.Properties.Resources.baseline_input_white_36dp;
            this.btnImportData.Location = new System.Drawing.Point(0, 165);
            this.btnImportData.Name = "btnImportData";
            this.btnImportData.Size = new System.Drawing.Size(230, 60);
            this.btnImportData.TabIndex = 4;
            this.btnImportData.Text = "Import Data";
            this.btnImportData.TextImageRelation = System.Windows.Forms.TextImageRelation.TextBeforeImage;
            this.btnImportData.UseVisualStyleBackColor = true;
            this.btnImportData.Click += new System.EventHandler(this.btnImportData_Click);
            this.btnImportData.Leave += new System.EventHandler(this.btnImportData_Leave);
            // 
            // panel2
            // 
            this.panel2.BackColor = System.Drawing.Color.DimGray;
            this.panel2.Controls.Add(this.pictureBox1);
            this.panel2.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel2.Location = new System.Drawing.Point(0, 0);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(230, 165);
            this.panel2.TabIndex = 1;
            // 
            // pictureBox1
            // 
            this.pictureBox1.Image = global::DAGR.Properties.Resources.dagger;
            this.pictureBox1.InitialImage = global::DAGR.Properties.Resources.dagger;
            this.pictureBox1.Location = new System.Drawing.Point(38, 9);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(153, 150);
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            // 
            // customTabControl1
            // 
            this.customTabControl1.Controls.Add(this.tabDashboard);
            this.customTabControl1.Controls.Add(this.tabAnalysis);
            this.customTabControl1.Controls.Add(this.tabImportData);
            this.customTabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.customTabControl1.Location = new System.Drawing.Point(230, 0);
            this.customTabControl1.Name = "customTabControl1";
            this.customTabControl1.SelectedIndex = 0;
            this.customTabControl1.Size = new System.Drawing.Size(1070, 795);
            this.customTabControl1.TabIndex = 3;
            // 
            // tabDashboard
            // 
            this.tabDashboard.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(28)))), ((int)(((byte)(28)))));
            this.tabDashboard.Location = new System.Drawing.Point(4, 22);
            this.tabDashboard.Name = "tabDashboard";
            this.tabDashboard.Padding = new System.Windows.Forms.Padding(3);
            this.tabDashboard.Size = new System.Drawing.Size(1062, 769);
            this.tabDashboard.TabIndex = 0;
            this.tabDashboard.Text = "Dashboard";
            // 
            // tabAnalysis
            // 
            this.tabAnalysis.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(28)))), ((int)(((byte)(28)))));
            this.tabAnalysis.Controls.Add(this.tableLayoutPanelMaster);
            this.tabAnalysis.Location = new System.Drawing.Point(4, 22);
            this.tabAnalysis.Name = "tabAnalysis";
            this.tabAnalysis.Padding = new System.Windows.Forms.Padding(3);
            this.tabAnalysis.Size = new System.Drawing.Size(1062, 769);
            this.tabAnalysis.TabIndex = 1;
            this.tabAnalysis.Text = "Analysis";
            // 
            // tableLayoutPanelMaster
            // 
            this.tableLayoutPanelMaster.ColumnCount = 2;
            this.tableLayoutPanelMaster.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanelMaster.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanelMaster.Controls.Add(this.tableLayoutPanelYaw, 1, 1);
            this.tableLayoutPanelMaster.Controls.Add(this.tableLayoutPanelRoll, 0, 1);
            this.tableLayoutPanelMaster.Controls.Add(this.tableLayoutPanelPitch, 1, 0);
            this.tableLayoutPanelMaster.Controls.Add(this.tableLayoutPanelAltitude, 0, 0);
            this.tableLayoutPanelMaster.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanelMaster.Location = new System.Drawing.Point(3, 3);
            this.tableLayoutPanelMaster.Name = "tableLayoutPanelMaster";
            this.tableLayoutPanelMaster.RowCount = 2;
            this.tableLayoutPanelMaster.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanelMaster.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanelMaster.Size = new System.Drawing.Size(1056, 763);
            this.tableLayoutPanelMaster.TabIndex = 11;
            // 
            // tableLayoutPanelYaw
            // 
            this.tableLayoutPanelYaw.ColumnCount = 1;
            this.tableLayoutPanelYaw.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanelYaw.Controls.Add(this.labelYaw, 0, 0);
            this.tableLayoutPanelYaw.Controls.Add(this.cartesianChartYaw, 0, 1);
            this.tableLayoutPanelYaw.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanelYaw.Location = new System.Drawing.Point(531, 384);
            this.tableLayoutPanelYaw.Name = "tableLayoutPanelYaw";
            this.tableLayoutPanelYaw.RowCount = 2;
            this.tableLayoutPanelYaw.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10F));
            this.tableLayoutPanelYaw.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 90F));
            this.tableLayoutPanelYaw.Size = new System.Drawing.Size(522, 376);
            this.tableLayoutPanelYaw.TabIndex = 12;
            // 
            // labelYaw
            // 
            this.labelYaw.AutoSize = true;
            this.labelYaw.Dock = System.Windows.Forms.DockStyle.Fill;
            this.labelYaw.Font = new System.Drawing.Font("Nirmala UI", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelYaw.ForeColor = System.Drawing.Color.White;
            this.labelYaw.Location = new System.Drawing.Point(3, 0);
            this.labelYaw.Name = "labelYaw";
            this.labelYaw.Size = new System.Drawing.Size(516, 37);
            this.labelYaw.TabIndex = 14;
            this.labelYaw.Text = "Yaw vs. Time";
            this.labelYaw.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // cartesianChartYaw
            // 
            this.cartesianChartYaw.Dock = System.Windows.Forms.DockStyle.Fill;
            this.cartesianChartYaw.Location = new System.Drawing.Point(3, 40);
            this.cartesianChartYaw.Name = "cartesianChartYaw";
            this.cartesianChartYaw.Size = new System.Drawing.Size(516, 333);
            this.cartesianChartYaw.TabIndex = 5;
            this.cartesianChartYaw.Text = "cartesianChart1";
            // 
            // tableLayoutPanelRoll
            // 
            this.tableLayoutPanelRoll.ColumnCount = 1;
            this.tableLayoutPanelRoll.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanelRoll.Controls.Add(this.labelRoll, 0, 0);
            this.tableLayoutPanelRoll.Controls.Add(this.cartesianChartRoll, 0, 1);
            this.tableLayoutPanelRoll.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanelRoll.Location = new System.Drawing.Point(3, 384);
            this.tableLayoutPanelRoll.Name = "tableLayoutPanelRoll";
            this.tableLayoutPanelRoll.RowCount = 2;
            this.tableLayoutPanelRoll.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10F));
            this.tableLayoutPanelRoll.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 90F));
            this.tableLayoutPanelRoll.Size = new System.Drawing.Size(522, 376);
            this.tableLayoutPanelRoll.TabIndex = 14;
            // 
            // labelRoll
            // 
            this.labelRoll.AutoSize = true;
            this.labelRoll.Dock = System.Windows.Forms.DockStyle.Top;
            this.labelRoll.Font = new System.Drawing.Font("Nirmala UI", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelRoll.ForeColor = System.Drawing.Color.White;
            this.labelRoll.Location = new System.Drawing.Point(3, 0);
            this.labelRoll.Name = "labelRoll";
            this.labelRoll.Size = new System.Drawing.Size(516, 30);
            this.labelRoll.TabIndex = 13;
            this.labelRoll.Text = "Roll vs. Time";
            this.labelRoll.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // cartesianChartRoll
            // 
            this.cartesianChartRoll.Dock = System.Windows.Forms.DockStyle.Fill;
            this.cartesianChartRoll.Location = new System.Drawing.Point(3, 40);
            this.cartesianChartRoll.Name = "cartesianChartRoll";
            this.cartesianChartRoll.Size = new System.Drawing.Size(516, 333);
            this.cartesianChartRoll.TabIndex = 3;
            this.cartesianChartRoll.Text = "cartesianChart1";
            // 
            // tableLayoutPanelPitch
            // 
            this.tableLayoutPanelPitch.ColumnCount = 1;
            this.tableLayoutPanelPitch.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanelPitch.Controls.Add(this.labelPitch, 0, 0);
            this.tableLayoutPanelPitch.Controls.Add(this.cartesianChartPitch, 0, 1);
            this.tableLayoutPanelPitch.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanelPitch.Location = new System.Drawing.Point(531, 3);
            this.tableLayoutPanelPitch.Name = "tableLayoutPanelPitch";
            this.tableLayoutPanelPitch.RowCount = 2;
            this.tableLayoutPanelPitch.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10F));
            this.tableLayoutPanelPitch.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 90F));
            this.tableLayoutPanelPitch.Size = new System.Drawing.Size(522, 375);
            this.tableLayoutPanelPitch.TabIndex = 13;
            // 
            // labelPitch
            // 
            this.labelPitch.AutoSize = true;
            this.labelPitch.Dock = System.Windows.Forms.DockStyle.Top;
            this.labelPitch.Font = new System.Drawing.Font("Nirmala UI", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelPitch.ForeColor = System.Drawing.Color.White;
            this.labelPitch.Location = new System.Drawing.Point(3, 0);
            this.labelPitch.Name = "labelPitch";
            this.labelPitch.Size = new System.Drawing.Size(516, 30);
            this.labelPitch.TabIndex = 13;
            this.labelPitch.Text = "Pitch vs. Time";
            this.labelPitch.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // cartesianChartPitch
            // 
            this.cartesianChartPitch.Dock = System.Windows.Forms.DockStyle.Fill;
            this.cartesianChartPitch.Location = new System.Drawing.Point(3, 40);
            this.cartesianChartPitch.Name = "cartesianChartPitch";
            this.cartesianChartPitch.Size = new System.Drawing.Size(516, 332);
            this.cartesianChartPitch.TabIndex = 4;
            this.cartesianChartPitch.Text = "cartesianChart1";
            // 
            // tableLayoutPanelAltitude
            // 
            this.tableLayoutPanelAltitude.ColumnCount = 1;
            this.tableLayoutPanelAltitude.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanelAltitude.Controls.Add(this.labelAltitude, 0, 0);
            this.tableLayoutPanelAltitude.Controls.Add(this.cartesianChartAltitude, 0, 1);
            this.tableLayoutPanelAltitude.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanelAltitude.Location = new System.Drawing.Point(3, 3);
            this.tableLayoutPanelAltitude.Name = "tableLayoutPanelAltitude";
            this.tableLayoutPanelAltitude.RowCount = 2;
            this.tableLayoutPanelAltitude.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10F));
            this.tableLayoutPanelAltitude.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 90F));
            this.tableLayoutPanelAltitude.Size = new System.Drawing.Size(522, 375);
            this.tableLayoutPanelAltitude.TabIndex = 12;
            // 
            // labelAltitude
            // 
            this.labelAltitude.AutoSize = true;
            this.labelAltitude.Dock = System.Windows.Forms.DockStyle.Top;
            this.labelAltitude.Font = new System.Drawing.Font("Nirmala UI", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelAltitude.ForeColor = System.Drawing.Color.White;
            this.labelAltitude.Location = new System.Drawing.Point(3, 0);
            this.labelAltitude.Name = "labelAltitude";
            this.labelAltitude.Size = new System.Drawing.Size(516, 30);
            this.labelAltitude.TabIndex = 13;
            this.labelAltitude.Text = "Altitude vs. Time";
            this.labelAltitude.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // cartesianChartAltitude
            // 
            this.cartesianChartAltitude.Dock = System.Windows.Forms.DockStyle.Fill;
            this.cartesianChartAltitude.Location = new System.Drawing.Point(3, 40);
            this.cartesianChartAltitude.Name = "cartesianChartAltitude";
            this.cartesianChartAltitude.Size = new System.Drawing.Size(516, 332);
            this.cartesianChartAltitude.TabIndex = 6;
            this.cartesianChartAltitude.Text = "cartesianChart1";
            // 
            // tabImportData
            // 
            this.tabImportData.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(28)))), ((int)(((byte)(28)))));
            this.tabImportData.Controls.Add(this.btnLoad);
            this.tabImportData.Controls.Add(this.tbImportFile);
            this.tabImportData.Controls.Add(this.btnBrowse);
            this.tabImportData.Location = new System.Drawing.Point(4, 22);
            this.tabImportData.Name = "tabImportData";
            this.tabImportData.Padding = new System.Windows.Forms.Padding(3);
            this.tabImportData.Size = new System.Drawing.Size(1062, 769);
            this.tabImportData.TabIndex = 2;
            this.tabImportData.Text = "Import Data";
            // 
            // btnLoad
            // 
            this.btnLoad.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnLoad.ForeColor = System.Drawing.Color.White;
            this.btnLoad.Location = new System.Drawing.Point(20, 61);
            this.btnLoad.Name = "btnLoad";
            this.btnLoad.Size = new System.Drawing.Size(75, 23);
            this.btnLoad.TabIndex = 2;
            this.btnLoad.Text = "Load";
            this.btnLoad.UseVisualStyleBackColor = true;
            this.btnLoad.Click += new System.EventHandler(this.btnLoad_Click);
            // 
            // tbImportFile
            // 
            this.tbImportFile.Location = new System.Drawing.Point(111, 24);
            this.tbImportFile.Name = "tbImportFile";
            this.tbImportFile.Size = new System.Drawing.Size(924, 20);
            this.tbImportFile.TabIndex = 1;
            // 
            // btnBrowse
            // 
            this.btnBrowse.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnBrowse.ForeColor = System.Drawing.SystemColors.ButtonHighlight;
            this.btnBrowse.Location = new System.Drawing.Point(20, 22);
            this.btnBrowse.Name = "btnBrowse";
            this.btnBrowse.Size = new System.Drawing.Size(75, 23);
            this.btnBrowse.TabIndex = 0;
            this.btnBrowse.Text = "Browse";
            this.btnBrowse.UseVisualStyleBackColor = true;
            this.btnBrowse.Click += new System.EventHandler(this.btnBrowse_Click);
            // 
            // DAGR
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(28)))), ((int)(((byte)(28)))));
            this.ClientSize = new System.Drawing.Size(1300, 795);
            this.Controls.Add(this.customTabControl1);
            this.Controls.Add(this.panel1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Name = "DAGR";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "DAGR";           
            this.panel1.ResumeLayout(false);
            this.panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.customTabControl1.ResumeLayout(false);
            this.tabAnalysis.ResumeLayout(false);
            this.tableLayoutPanelMaster.ResumeLayout(false);
            this.tableLayoutPanelYaw.ResumeLayout(false);
            this.tableLayoutPanelYaw.PerformLayout();
            this.tableLayoutPanelRoll.ResumeLayout(false);
            this.tableLayoutPanelRoll.PerformLayout();
            this.tableLayoutPanelPitch.ResumeLayout(false);
            this.tableLayoutPanelPitch.PerformLayout();
            this.tableLayoutPanelAltitude.ResumeLayout(false);
            this.tableLayoutPanelAltitude.PerformLayout();
            this.tabImportData.ResumeLayout(false);
            this.tabImportData.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Button btnDashboard;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Button btnSettings;
        private System.Windows.Forms.Button btnImportData;
        private System.Windows.Forms.Button btnAnalysis;
        private System.Windows.Forms.Panel pnlNav;
        private System.Windows.Forms.PictureBox pictureBox1;
        private CustomTabControl customTabControl1;
        private System.Windows.Forms.TabPage tabDashboard;
        private System.Windows.Forms.TabPage tabAnalysis;
        private System.Windows.Forms.TabPage tabImportData;
        private LiveCharts.WinForms.CartesianChart cartesianChartRoll;
        private LiveCharts.WinForms.CartesianChart cartesianChartPitch;
        private LiveCharts.WinForms.CartesianChart cartesianChartYaw;
        private System.Windows.Forms.TextBox tbImportFile;
        private System.Windows.Forms.Button btnBrowse;
        private System.Windows.Forms.Button btnLoad;
        private LiveCharts.WinForms.CartesianChart cartesianChartAltitude;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanelMaster;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanelPitch;
        private System.Windows.Forms.Label labelPitch;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanelAltitude;
        private System.Windows.Forms.Label labelAltitude;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanelRoll;
        private System.Windows.Forms.Label labelRoll;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanelYaw;
        private System.Windows.Forms.Label labelYaw;
    }
}

