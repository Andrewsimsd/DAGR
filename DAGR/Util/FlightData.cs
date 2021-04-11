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
        public float Altitude { get; set; }

        [Name("pitch")]
        public float Pitch { get; set; }

        [Name("roll")]
        public float Roll { get; set; }

        [Name("yaw")]
        public float Yaw { get; set; }

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
            Altitude = float.Parse(alt);
            Pitch = float.Parse(pit);
            Roll = float.Parse(roll);
            Yaw = float.Parse(yaw);
        }
        /// <summary>
        /// Float Converted constructor
        /// </summary>
        public FlightData(float alt, float pit, float roll, float yaw)
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
