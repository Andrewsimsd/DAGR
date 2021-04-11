using System;
using System.Collections.Generic;
using System.IO;

namespace DataUtil
{
    public class FlightData
    {
        /// <summary>
        /// Data memebrs
        /// </summary>
        public List<float> altitude;
        public List<float> pitch;
        public List<float> roll;
        public List<float> yaw;

        /// <summary>
        /// Default constructor
        /// </summary>
        public FlightData()
        {
            altitude = new List<float>();
            pitch = new List<float>();
            roll = new List<float>();
            yaw = new List<float>();
        }
        public void ReadCsv(string filePath)
        {
            StreamReader reader = null;
            if (File.Exists(filePath))
            {
                reader = new StreamReader(File.OpenRead(filePath));
                string headerLine = reader.ReadLine();
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    this.altitude.Add(float.Parse(values[0]));
                    this.pitch.Add(float.Parse(values[1]));
                    this.roll.Add(float.Parse(values[2]));
                    this.yaw.Add(float.Parse(values[3]));
                }
            }
            else
            {
                Console.WriteLine("File doesn't exist");
            }
        }
    }
}
