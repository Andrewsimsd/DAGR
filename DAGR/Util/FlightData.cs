using System;
using System.Collections.Generic;
using System.IO;
using CsvHelper;
using CsvHelper.Configuration.Attributes;

namespace DataUtil
{
    public class FlightData
    {
        /// <summary>
        /// Data members
        /// </summary>
        [Name("altitude")]
        public double Altitude { get; set; }

        [Name("pitch")]
        public double Pitch { get; set; }

        [Name("roll")]
        public double Roll { get; set; }

        [Name("yaw")]
        public double Yaw { get; set; }

        /// <summary>
        /// Default constructor
        /// </summary>
        public FlightData()
        {
            Altitude = 0;
            Pitch = 0;
            Roll = 0;
            Yaw = 0;
        }
        /// <summary>
        /// String constructor
        /// </summary>
        public FlightData(string alt, string pit, string roll, string yaw)
        {
            Altitude = double.Parse(alt);
            Pitch = double.Parse(pit);
            Roll = double.Parse(roll);
            Yaw = double.Parse(yaw);
        }
        /// <summary>
        /// Float Converted constructor
        /// </summary>
        public FlightData(double alt, double pit, double roll, double yaw)
        {
            Altitude = alt;
            Pitch = pit;
            Roll = roll;
            Yaw = yaw;
        }

        public override string ToString() 
        {
            return $"Altitude:{Altitude}, Pitch:{Pitch}, Roll:{Roll}, Yaw:{Yaw}";
        }
       
    }
}
