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
            this.components = new System.ComponentModel.Container();
            this.panel1 = new System.Windows.Forms.Panel();
            this.pnlNav = new System.Windows.Forms.Panel();
            this.btnSettings = new System.Windows.Forms.Button();
            this.btnAnalysis = new System.Windows.Forms.Button();
            this.btnDashboard = new System.Windows.Forms.Button();
            this.btnImportData = new System.Windows.Forms.Button();
            this.panel2 = new System.Windows.Forms.Panel();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.customTabControl1 = new DAGR.CustomTabControl();
            this.tabDashboard = new System.Windows.Forms.TabPage();
            this.tabAnalysis = new System.Windows.Forms.TabPage();
            this.tableLayoutPanelMaster = new System.Windows.Forms.TableLayoutPanel();
            this.tabImportData = new System.Windows.Forms.TabPage();
            this.btnLoad = new System.Windows.Forms.Button();
            this.tbImportFile = new System.Windows.Forms.TextBox();
            this.btnBrowse = new System.Windows.Forms.Button();
            this.formsPlotAltVsTime = new ScottPlot.FormsPlot();
            this.formsPlotPitchVsTime = new ScottPlot.FormsPlot();
            this.formsPlotRollVsTime = new ScottPlot.FormsPlot();
            this.formsPlotYawVsTime = new ScottPlot.FormsPlot();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.customTabControl1.SuspendLayout();
            this.tabAnalysis.SuspendLayout();
            this.tableLayoutPanelMaster.SuspendLayout();
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
            this.tableLayoutPanelMaster.Controls.Add(this.formsPlotYawVsTime, 1, 1);
            this.tableLayoutPanelMaster.Controls.Add(this.formsPlotRollVsTime, 0, 1);
            this.tableLayoutPanelMaster.Controls.Add(this.formsPlotPitchVsTime, 1, 0);
            this.tableLayoutPanelMaster.Controls.Add(this.formsPlotAltVsTime, 0, 0);
            this.tableLayoutPanelMaster.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanelMaster.Location = new System.Drawing.Point(3, 3);
            this.tableLayoutPanelMaster.Name = "tableLayoutPanelMaster";
            this.tableLayoutPanelMaster.RowCount = 2;
            this.tableLayoutPanelMaster.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanelMaster.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanelMaster.Size = new System.Drawing.Size(1056, 763);
            this.tableLayoutPanelMaster.TabIndex = 11;
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
            // formsPlotAltVsTime
            // 
            this.formsPlotAltVsTime.Location = new System.Drawing.Point(3, 3);
            this.formsPlotAltVsTime.Name = "formsPlotAltVsTime";
            this.formsPlotAltVsTime.Size = new System.Drawing.Size(522, 374);
            this.formsPlotAltVsTime.TabIndex = 0;
            // 
            // formsPlotPitchVsTime
            // 
            this.formsPlotPitchVsTime.Location = new System.Drawing.Point(531, 3);
            this.formsPlotPitchVsTime.Name = "formsPlotPitchVsTime";
            this.formsPlotPitchVsTime.Size = new System.Drawing.Size(522, 374);
            this.formsPlotPitchVsTime.TabIndex = 1;
            // 
            // formsPlotRollVsTime
            // 
            this.formsPlotRollVsTime.Location = new System.Drawing.Point(3, 384);
            this.formsPlotRollVsTime.Name = "formsPlotRollVsTime";
            this.formsPlotRollVsTime.Size = new System.Drawing.Size(522, 374);
            this.formsPlotRollVsTime.TabIndex = 2;
            // 
            // formsPlotYawVsTime
            // 
            this.formsPlotYawVsTime.Location = new System.Drawing.Point(531, 384);
            this.formsPlotYawVsTime.Name = "formsPlotYawVsTime";
            this.formsPlotYawVsTime.Size = new System.Drawing.Size(522, 374);
            this.formsPlotYawVsTime.TabIndex = 3;
            // 
            // DashboardForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(28)))), ((int)(((byte)(28)))), ((int)(((byte)(28)))));
            this.ClientSize = new System.Drawing.Size(1300, 795);
            this.Controls.Add(this.customTabControl1);
            this.Controls.Add(this.panel1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Name = "DashboardForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "DAGR";
            this.panel1.ResumeLayout(false);
            this.panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.customTabControl1.ResumeLayout(false);
            this.tabAnalysis.ResumeLayout(false);
            this.tableLayoutPanelMaster.ResumeLayout(false);
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
        private System.Windows.Forms.TextBox tbImportFile;
        private System.Windows.Forms.Button btnBrowse;
        private System.Windows.Forms.Button btnLoad;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanelMaster;
        private ScottPlot.FormsPlot formsPlotYawVsTime;
        private ScottPlot.FormsPlot formsPlotRollVsTime;
        private ScottPlot.FormsPlot formsPlotPitchVsTime;
        private ScottPlot.FormsPlot formsPlotAltVsTime;
    }
}

