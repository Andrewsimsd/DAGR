using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using CsvHelper;

namespace DataUtil
{
    public class DAGRReader
    {
        public static List<FlightData> ReadFlightData(string filePath)
        {            
            List<FlightData> flightDataList = new List<FlightData>();

            if (File.Exists(filePath))
            {
                var config = new CsvHelper.Configuration.CsvConfiguration(CultureInfo.InvariantCulture)
                {
                    PrepareHeaderForMatch = args => args.Header.ToLower(),
                };
                using (var reader = new StreamReader(filePath))
                using (var csv = new CsvReader(reader, config))
                {
                    flightDataList = csv.GetRecords<FlightData>().ToList();
                }
            }
            else
            {
                //TODO log and handle error since return of function is empty list
                Console.WriteLine("File doesn't exist");                
            }
            return flightDataList;
        }
    }
}
