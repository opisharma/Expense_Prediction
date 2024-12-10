#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_USERS 100
#define MAX_TRANSACTIONS 365
#define MAX_LINE 1024
#define LEARNING_RATE 0.001
#define ITERATIONS 1000

typedef enum {
    WITHDRAWAL = 1,
    DEPOSIT = -1,
    TRANSFER = 0
} TransactionType;

typedef struct {
    char username[50];
    double expenses[MAX_TRANSACTIONS];
    int days[MAX_TRANSACTIONS];
    int recordCount;
    double m;
    double b;
} UserExpense;

void gradientDescent(UserExpense *user) {
    double m = 0.0, b = 0.0;
    int n = user->recordCount;

    if (n == 0) {
        printf("No data for user %s. Skipping gradient descent.\n", user->username);
        return;
    }

    for (int iter = 0; iter < ITERATIONS; iter++) {
        double gradient_m = 0.0, gradient_b = 0.0;

        for (int i = 0; i < n; i++) {
            double x = user->days[i];
            double y = user->expenses[i];
            double prediction = m * x + b;
            gradient_m += -2 * x * (y - prediction);
            gradient_b += -2 * (y - prediction);
        }

        // Update m and b
        m -= LEARNING_RATE * gradient_m / n;
        b -= LEARNING_RATE * gradient_b / n;
    }

    user->m = m;
    user->b = b;
}

void predictNext7Days(UserExpense *user) {
    printf("Predicted expenses for user %s:\n", user->username);
    for (int i = 1; i <= 7; i++) {
        int nextDay = (user->recordCount + i > 31) ? (user->recordCount + i) % 31 : (user->recordCount + i);
        double prediction = user->m * nextDay + user->b;
        printf("Day %d: %.2f\n", nextDay, prediction);
    }
}

TransactionType parseTransactionType(const char *transactionTypeStr) {
    if (strcmp(transactionTypeStr, "Withdrawal") == 0) {
        return WITHDRAWAL;
    } else if (strcmp(transactionTypeStr, "Deposit") == 0) {
        return DEPOSIT;
    } else if (strcmp(transactionTypeStr, "Transfer") == 0) {
        return TRANSFER;
    }
    return 0;
}

double convertAmount(const char *amountStr) {
    char cleanAmount[50];
    int j = 0;

    for (int i = 0; amountStr[i] != '\0'; i++) {
        if (amountStr[i] != '"' && amountStr[i] != ',') {
            cleanAmount[j++] = amountStr[i];
        }
    }
    cleanAmount[j] = '\0';

    return atof(cleanAmount);
}

int main() {
    FILE *file = fopen("transactions.csv", "r");
    if (!file) {
        perror("Unable to open file");
        return 1;
    }

    UserExpense users[MAX_USERS];
    int userCount = 0;

    char line[MAX_LINE];
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file)) {
        char transactionTypeStr[50], username[50], transactionAmountStr[50];
        double transactionAmount;
        int dayX;  // Day of the month (1-31)
        TransactionType transactionType;

        sscanf(line, "%[^,],%[^,],%[^,],%*[^,],%*[^,],%d,%*[^,],%*[^,],%[^,\n]",
               transactionTypeStr, username, transactionAmountStr, &dayX, transactionAmountStr);

        transactionType = parseTransactionType(transactionTypeStr);

        if (transactionType != 0 && dayX >= 1 && dayX <= 31) {
            transactionAmount = convertAmount(transactionAmountStr);

            int found = 0;
            for (int i = 0; i < userCount; i++) {
                if (strcmp(users[i].username, username) == 0) {
                    users[i].expenses[users[i].recordCount] = transactionAmount;
                    users[i].days[users[i].recordCount] = dayX;
                    users[i].recordCount++;
                    found = 1;
                    break;
                }
            }
            if (!found && userCount < MAX_USERS) {
                strcpy(users[userCount].username, username);
                users[userCount].expenses[0] = transactionAmount;
                users[userCount].days[0] = dayX;
                users[userCount].recordCount = 1;
                userCount++;
            }
        }
    }

    fclose(file);

    printf("Transactions data:\n");
    for (int i = 0; i < userCount; i++) {
        printf("User: %s\n", users[i].username);
        for (int j = 0; j < users[i].recordCount; j++) {
            printf("  Day %d: Amount %.2f\n", users[i].days[j], users[i].expenses[j]);
        }
    }

    for (int i = 0; i < userCount; i++) {
        gradientDescent(&users[i]);
        predictNext7Days(&users[i]);
    }

    return 0;
}