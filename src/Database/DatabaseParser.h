#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Parser for healthcare dataset

/**
 * @brief A structure to represent the healthcare dataset.
 * Each field corresponds to a column in the dataset.
 */
struct HealthCare {
    std::vector<std::string> Name;
    std::vector<int> Age;
    std::vector<std::string> Gender;
    std::vector<std::string> BloodType;
    std::vector<std::string> MedicalCondition;
    std::vector<std::string> DateOfAdmission;
    std::vector<std::string> Doctor;
    std::vector<std::string> Hospital;
    std::vector<std::string> InsuranceProvider;
    std::vector<float> BillingAmount;
    std::vector<int> RoomNumber;
    std::vector<std::string> AdmissionType;
    std::vector<std::string> DischargeDate;
    std::vector<std::string> Medication;
    std::vector<std::string> TestResults;
};

/**
 * @brief Removes leading and trailing whitespaces from a string.
 * 
 * @param str Input string to be trimmed.
 * @return Trimmed string.
 */
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    size_t last = str.find_last_not_of(" \t\n\r");
    return (first == std::string::npos || last == std::string::npos) ? "" : str.substr(first, (last - first + 1));
}

/**
 * @brief Parses a field from a CSV line. Handles quoted and unquoted fields.
 * 
 * @param ss The stringstream representing the current line of the CSV file.
 * @return The parsed and trimmed field value.
 */
std::string parseField(std::stringstream& ss) {
    std::string token;
    char ch;

    while (ss.peek() == ' ' || ss.peek() == '\t' || ss.peek() == '\n') {
        ss.get(); // Skip leading whitespaces.
    }

    if (ss.peek() == '"') {
        ss.get(); // Skip the opening quote.
        while (ss.get(ch)) {
            if (ch == '"') {
                if (ss.peek() == '"') {
                    ss.get(); // Handle escaped double quotes.
                    token += '"';
                } else {
                    break; // End of quoted field.
                }
            } else {
                token += ch;
            }
        }
        if (ss.peek() == ',') {
            ss.get(); // Skip the delimiter.
        }
    } else {
        std::getline(ss, token, ','); // Parse unquoted field.
    }

    return trim(token);
}

/**
 * @brief Checks whether a given string represents a valid float value.
 * 
 * @param str The input string.
 * @return True if the string can be converted to a float, false otherwise.
 */
bool isValidFloat(const std::string& str) {
    try {
        std::stof(str);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::out_of_range&) {
        return false;
    }
}

/**
 * @brief Processes a CSV file to populate the HealthCare dataset structure.
 * 
 * @param filePath The path to the CSV file.
 * @param records The HealthCare structure to store the data.
 * @param maxLines The maximum number of lines to process from the CSV.
 */
void processCSV(const std::string &filePath, HealthCare &records, int maxLines = 100000) {
    std::ifstream file(filePath);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return;
    }

    // Skip the header line.
    std::getline(file, line);

    int lines_t = 0;

    // Process each line in the CSV file.
    while (std::getline(file, line) && lines_t < maxLines) {
        ++lines_t;

        std::stringstream ss(line);
        std::string token;

        // Parse and store each field in the corresponding column vector.
        records.Name.push_back(parseField(ss));
        records.Age.push_back(std::stoi(parseField(ss)));
        records.Gender.push_back(parseField(ss));
        records.BloodType.push_back(parseField(ss));
        records.MedicalCondition.push_back(parseField(ss));
        records.DateOfAdmission.push_back(parseField(ss));
        records.Doctor.push_back(parseField(ss));
        records.Hospital.push_back(parseField(ss));
        records.InsuranceProvider.push_back(parseField(ss));

        token = parseField(ss);
        if (isValidFloat(token)) {
            records.BillingAmount.push_back(std::fabs(std::stof(token)));
        } else {
            if (lines_t < 10) {
                std::cerr << "Line: " << lines_t << ", Invalid billing amount: " << token << std::endl;
            }
            records.BillingAmount.push_back(0.0f); // Assign a default value for invalid data.
        }

        records.RoomNumber.push_back(std::stoi(parseField(ss)));
        records.AdmissionType.push_back(parseField(ss));
        records.DischargeDate.push_back(parseField(ss));
        records.Medication.push_back(parseField(ss));
        records.TestResults.push_back(parseField(ss));
    }

    file.close();
}

/**
 * @brief Displays the first 10 rows of the HealthCare dataset for verification.
 * 
 * @param records The HealthCare dataset structure.
 */
void displayData(const HealthCare &records) {
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "Name: " << records.Name[i]
                  << ", Age: " << records.Age[i]
                  << ", Gender: " << records.Gender[i]
                  << ", Blood Type: " << records.BloodType[i]
                  << ", Medical Condition: " << records.MedicalCondition[i]
                  << ", Date of Admission: " << records.DateOfAdmission[i]
                  << ", Doctor: " << records.Doctor[i]
                  << ", Hospital: " << records.Hospital[i]
                  << ", Insurance Provider: " << records.InsuranceProvider[i]
                  << ", Billing Amount: " << records.BillingAmount[i]
                  << ", Room Number: " << records.RoomNumber[i]
                  << ", Admission Type: " << records.AdmissionType[i]
                  << ", Discharge Date: " << records.DischargeDate[i]
                  << ", Medication: " << records.Medication[i]
                  << ", Test Results: " << records.TestResults[i]
                  << std::endl;
    }
}

/**
 * @brief Encodes a column of strings into integer codes.
 * 
 * @param column The input column of strings.
 * @param EncodedColumn The output column of encoded integers.
 * @param Code2Element The mapping of integer codes to original strings.
 * @param elementToCode The mapping of strings to integer codes.
 */
void Encode(const std::vector<std::string>& column, std::vector<int>& EncodedColumn, std::vector<std::string>& Code2Element, std::unordered_map<std::string, int>& elementToCode) {
    int currentCode = 0;
    for (const auto& element : column) {
        if (elementToCode.find(element) == elementToCode.end()) {
            elementToCode[element] = currentCode;
            Code2Element.push_back(element); 
            currentCode++;
        }
        EncodedColumn.push_back(elementToCode[element]); 
    }
}

/**
 * @brief Converts a date string to an integer representation (YYYYMMDD).
 * 
 * @param dateStr The input date string.
 * @return The integer representation of the date.
 */
int convertDateToInt(const std::string& dateStr) {
    int year, month, day;
    char delimiter1, delimiter2;

    std::stringstream ss(dateStr);
    ss >> year >> delimiter1 >> month >> delimiter2 >> day;

    return year * 10000 + month * 100 + day;
}

/**
 * @brief Processes date strings into integer representations.
 * 
 * @param Date The input vector of date strings.
 * @param iDate The output vector of integer date representations.
 */
void processDate(std::vector<std::string> &Date, std::vector<int> &iDate) {
    iDate.resize(Date.size());
    for (size_t i = 0; i < Date.size(); ++i) {
        iDate[i] = convertDateToInt(Date[i]);
    }
}

/**
 * @brief Generates a word-wise representation of substrings for a "LIKE" pattern match.
 * 
 * This function takes an input vector of strings, generates substrings of a specified pattern size,
 * and encodes these substrings into integer representations. The encoded substrings are stored in 
 * a two-dimensional output vector.
 * 
 * @param input The input vector of strings for which substrings need to be generated.
 * @param output The output two-dimensional vector where each row contains encoded substrings for the corresponding input string.
 * @param pattern The "LIKE" pattern to determine the size of substrings (e.g., 'ppl' would generate substrings of size 3).
 * @param elementToCode A mapping from substrings to their corresponding integer codes (used for encoding).
 * @param Code2Element A mapping from integer codes to their corresponding substrings (used for decoding).
 */
void genLike(std::vector<std::string> &input, 
             std::vector<std::vector<int>> &output, 
             std::string &pattern, 
             std::unordered_map<std::string, int>& elementToCode, 
             std::vector<std::string>& Code2Element) {
    int pattern_size = pattern.size();  // The size of the substrings to generate (based on the pattern length).
    int currentCode = 0;  // Tracks the next available code for encoding new substrings.

    // Resize the output to match the size of the input vector.
    output.resize(input.size());

    // Iterate over each string in the input vector.
    for (size_t i = 0; i < input.size(); ++i) {
        std::string element = input[i];
        output[i].clear();  // Clear the output row for the current string.

        // Generate substrings of the specified pattern size from the current string.
        for (size_t j = 0; j <= element.size() - pattern_size; ++j) {
            std::string temp = element.substr(j, pattern_size);  // Extract a substring of size 'pattern_size' starting at index 'j'.

            // If the substring is not yet encoded, assign it a new integer code.
            if (elementToCode.find(temp) == elementToCode.end()) {
                elementToCode[temp] = currentCode;  // Map the substring to the next available integer code.
                Code2Element.push_back(temp);  // Store the substring in the decoding vector.
                currentCode++;  // Increment the code for the next unique substring.
            }

            // Add the encoded integer value of the substring to the output.
            output[i].push_back(elementToCode[temp]);
        }
    }
}

/**
 * @brief Check that insurance matches pattern
 */
bool is_match(const std::string& insurance, const std::string& pattern) {
    size_t ins_len = insurance.length();
    size_t pat_len = pattern.length();
    
    if (ins_len < pat_len) {
        return false;
    }

    for (size_t i = 0; i <= ins_len - pat_len; ++i) {
        bool match = true;
        for (size_t j = 0; j < pat_len; ++j) {
            if (insurance[i + j] != pattern[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}


template <class P>
void multiply(TFHEpp::TLWE<TFHEpp::lvl1param> &LWE1, TFHEpp::TLWE<P> &LWE2, TFHEpp::TLWE<P> &LWE, TFHEpp::EvalKey &ek, TFHEpp::SecretKey &sk)
{
    for (int k = 0; k <= P::n; ++k)
    {
        LWE[k] = LWE2[k];
    }
    if (!TFHEpp::tlweSymDecrypt<TFHEpp::lvl1param>(LWE1, sk.key.get<TFHEpp::lvl1param>()))
    {
        for (int k = 0; k <= P::n; ++k)
        {
            LWE[k] -= LWE2[k];
        }
    }
}