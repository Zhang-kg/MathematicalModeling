#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define INIT_TYPE 10
#define ALLTOONE_TYPE 100
#define ONETOALL_TYPE 200
#define MULTI_TYPE 300
#define RESULT_TYPE 400
#define RESULT_LEN 500
#define MULTI_LEN 600

int split;
long DataSize;
int *data, *buffer;
int mylength;
int* index;
int *samplePoints;

void Psrs_Main();
void merror(char *ch);
void sswap(int* a, int* b);
int NumberOfThree(int arr[], int low, int high);
int Partition(int a[], int p, int r);
void InsertSort(int arr[], int m, int n);
void qqqsot(int arr[],int low,int high);
void quicksort(int *datas, int bb, int ee);
void merge(const int src[], int dst[], int l, int mid, int r);
void subArrayMerge(int src[], int buf[], int index[], int l, int r);
void multimerge(int src[], int buf[], const int length[], int total);

int main(int argc, char *argv[])
{
    long BaseNum = 32;
    int PlusNum;
    int MyID;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);

    PlusNum = 60;
    DataSize = 1024;

    if (MyID == 0)
        printf("The DataSize is : %lu\n", DataSize);
    Psrs_Main();
    MPI_Barrier(MPI_COMM_WORLD);
    if (MyID == 0)
        printf("\n");

    MPI_Finalize();
}

// * 并行正则采样排序主体函数
void Psrs_Main() {


    // * 变量声明
    // * i和j为循环变量，用在循环中
    int i, j;
    // * MyID表示当前进程号，SumID表示总进程数
    int MyID, SumID;
    int n, k, l;
    FILE *fp;
    int ready;
    MPI_Status status[32 * 32 * 2];
    MPI_Request request[32 * 32 * 2];

    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
    MPI_Comm_size(MPI_COMM_WORLD, &SumID);
    if (MyID == 0) {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer [128];

        time (&rawtime);
        // printf("%ld\n", rawtime);

        timeinfo = localtime (&rawtime);
        strftime(buffer,sizeof(buffer),"Now is %Y/%m/%d %H:%M:%S",timeinfo);
        printf("%s\n", buffer);
    }
    /*
    * 分配内存空间，并进行初始化
    * split为分段点数量，在合并时各个处理器要将其负责的一段
        数据分成p段，这个对应从中选择的p-1个分段点
    * mylength 表示某个处理器需要处理的数据量。总共有Datasize个数据，
        处理器的个数时SumId，因此Datasize / SumID得到每个处理器需要处理的数据量
    * data数组是长度为Datasize的数组，表示原始数据
    * buffer数组是长度为Datasize的数组，用作缓冲区
    * 注：这里一次性使用malloc申请了2*Datasize的数组大小，
        主要原因是避免归并时重复malloc出buffer造成的性能下降。
    ! 之后需要将buffer使用的地方进行说明
    */
    split = SumID - 1;
    mylength = DataSize / SumID;
    data = (int *)malloc(2 * DataSize * sizeof(int));
    if (data == 0)
        merror("malloc memory for data error!");
    buffer = &data[DataSize];

    if (SumID > 1) {
        /*
        * samplePoints：这里分配的samplePoints表示所有进程发送来的样本点
            大小计算：每个进程从其处理的数据中选择split-1个样本点，共有SumID个进程
            因此总数为SumID * split个样本点
            在后续的程序中，不同进程发送来的样本点会被存放到samplePoints中不同的位置
        * index：这里分配的index表示该进程中根据主元进行划分后对应的每段的起始和终止
            位置。例如某个进程Pi将它处理的元素分成p段后，第i段的起始位置为index[i*x]
            终止位置为index[2*i+1]
        */
        samplePoints = (int *)malloc(sizeof(int) * SumID * split);
        if (samplePoints == 0)
            merror("malloc memory for samplePoints error!");
        index = (int *)malloc(sizeof(int) * 2 * SumID);
        if (index == 0)
            merror("malloc memory for index error!");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ? 使用MPI_Send和MPI_Recv将输出串行化
    /*
    * mutex为一个内容无意义的东西，但是它的存在的作用相当于一个锁，
        只有拿到这个锁的进程才能输出。
    * 具体方法和第一次作业的ring一样，当某个进程想要输出时，首先检查
        上个进程（特指MyID-1的进程）是否输出完毕。只有上个进程输出完毕
        后才会释放锁给当前线程，让其进行输出。当前线程输出完毕后，才会释放锁给
        下个进程。如此实现了输出的有序
    */
    int mutex;
    if (MyID != 0) {
        // * 当前进程准备输出，向上个进程索要锁（mutex）
        MPI_Recv(&mutex, 1, MPI_CHAR, MyID - 1, ONETOALL_TYPE, MPI_COMM_WORLD, status);
    }
    /*
    * 用自身id作为随机数的种子，确保每次执行时生成的数据相同，便于调试
    */
    srand(MyID);

    printf("This is node %d \n", MyID);
    printf("On node %d the input data is:\n", MyID);
    for (i = 0; i < mylength; i++) {
        /*
        * 数据是在各个处理器上使用随机数生成的
        * 这里对源程序进行了修改，运行源程序可以发现随机数生成的数字太大。
            进行取模并减去0x8000的作用是减小生成的数字
        */
        data[i] = (int)random() % 0xffff - 0x8000;
        printf("%d : ", data[i]);
    }
    printf("\n");

    if (MyID != SumID - 1) {
        // * 当前进程输出完毕，将锁（mutex）交给下个进程让其输出
        MPI_Send(&mutex, 1, MPI_CHAR, MyID + 1, ONETOALL_TYPE, MPI_COMM_WORLD);
    }

    /*
    * 每个处理器将自己的n/P个数据用串行快速排序(Quicksort)，
        得到一个排好序的序列，对应于算法13.5步骤（1）
    ! 源程序中在快排前还设置了一次MPI_Barrier，在经过输出的约束处理之后，
        可能是为了保证所有进程均输出完毕之后再统一进行排序处理。
        这里可以取消第一次等待，将等待时机保证所有进程都经过排序即可
    */
    // ! quicksort(data, 0, mylength - 1);
    qqqsot(data, 0, mylength - 1);
    MPI_Barrier(MPI_COMM_WORLD);

    /*
    * 每个处理器从排好序的序列中选取第w，2w，3w，…，(P-1)w个共P-1个数据
        作为代表元素，其中w=n/P*P，对应于算法13.5步骤（2）
    */
    if (SumID > 1) {
        // MPI_Barrier(MPI_COMM_WORLD);
        // * 这里进行代表元素的选择
        // * 其中n表示两个代表元素之间间隔元素个数
        n = (int)(mylength / (split + 1));
        // * 将选择得到的元素放到samplePoints中
        for (i = 0; i < split; i++)
            samplePoints[i] = data[(i + 1) * n - 1];

        MPI_Barrier(MPI_COMM_WORLD);

        if (MyID == 0) {
            /*
            * 每个处理器将选好的代表元素送到处理器P0中，
                对应于算法13.5步骤（3）
            */
            j = 0;
            for (i = 1; i < SumID; i++) {
                /*
                * MPI_Irecv是非阻塞式接受，由于每个进将代表元素发送给0好进程
                    中的samplePoints的不同位置，所以可以同步进行接受
                    MPI_Irecv的接口定义：
                    int MPIAPI MPI_Irecv(
                    _In_opt_ void         *buf,
                             int          count,
                             MPI_Datatype datatype,
                             int          source,
                             int          tag,
                             MPI_Comm     comm,
                    _Out_    MPI_Request  *request
                    );
                */
                MPI_Irecv(
                    &samplePoints[i * split],   // * 指向包含要发送的数据缓冲区的指针
                    sizeof(int) * split,        // * 缓冲区数组中的元素数。 如果消息的数据部分为空，请将 count 参数设置为 0
                    MPI_CHAR,                   // * 缓冲区中元素的数据类型。
                    i,                          // * 指定通信器内的发送进程的排名。 指定 MPI_ANY_SOURCE 常量以指定任何源是可接受的。
                    ALLTOONE_TYPE + i,          // * 可用于区分不同类型的消息的消息标记
                    MPI_COMM_WORLD,             // * 通信器的句柄
                    &request[j++]);             // * 返回时，包含所请求通信操作的句柄
            }
            /*
            * 等待所有数据完成操作，接口定义为：
                int MPIAPI MPI_Waitall(
                    int                              count,
                    _Inout_count_(count) MPI_Request *array_of_requests,
                    _Out_cap_(count) MPI_Status      *array_of_statuses
                );
            */
            MPI_Waitall(
                SumID - 1,  // * count: array_of_requests 参数中的条目数
                request,    // * array_of_requests: 未完成操作的 MPI_Request 句柄数组
                status      // * array_of_statuses: 描述已完成操作的MPI_Status数组
            );
            // * 这里进行第一次同步，表示所有数据发送完成
            MPI_Barrier(MPI_COMM_WORLD);
            /*
            * 处理器P0将上一步送来的P段有序的数据序列做P路归并，
                再选择排序后的第P-1，2(P-1)，…，(P-1)(P-1)个共P-1个主元，
                对应于算法13.5步骤（3）
            * 这里使用串行快速排序，快速排序和P路归并排序的时间复杂度均为O(nlogn)
                但是如果使用归并排序，则还需要使用额外辅助空间，快排可以直接在原数组上进行
            */
            //! quicksort(samplePoints, 0, SumID * split - 1);
            // if (MyID == 0) {
            //     printf("\n\n\n\nnode 0 before qqqsort\n\n");
            //     for (i = 0; i <= SumID * split - 1; i++) {
            //         printf("%d, ", samplePoints[i]);
            //     }
            //     printf("\n");
            // }
            // quicksort(samplePoints, 0, SumID * split - 1);
            qqqsot(samplePoints, 0, SumID * split - 1);
            // if (MyID == 0) {
            //     printf("\n\n\n\nnode 0 after qqqsort\n\n");
            //     for (i = 0; i <= SumID * split - 1; i++) {
            //         printf("%d, ", samplePoints[i]);
            //     }
            //     printf("\n");
            // }
            // * 这里进行第二次排序，表示0号进程完成排序
            MPI_Barrier(MPI_COMM_WORLD);
            /*
            * 此时samplePoints中保存着所有代表元素经过排序后的结果，根据算法思想
                将从所有代表元素中选择p-1个主元，这些主元将发送到其他进程中，供其他进程
                进行划分元素，发送数据段
            */
            for (i = 1; i <= split; i++) {
                // * 这里将所有主元集中在samplePoints[0~split]
                samplePoints[i] = samplePoints[i * split - 1];
            }

            /*
            * 处理器P0将这P-1个主元播送到所有处理器中，对应于算法13.5步骤（4）
                MPI_Bcast接口定义为：
                int MPIAPI MPI_Bcast(
                    _Inout_  void        *buffer,
                    _In_    int          count,
                    _In_    MPI_Datatype datatype,
                    _In_    int          root,
                    _In_    MPI_Comm     comm
                );
            */
            MPI_Bcast(
                samplePoints,               // * 指向数据缓冲区的指针。
                sizeof(int) * (1 + split),  // * 缓冲区中的数据元素数
                MPI_CHAR,                   // * 发送缓冲区中元素的 MPI 数据类型
                0,                          // * 正在发送数据的进程的id
                MPI_COMM_WORLD              // * MPI_Comm通信器句柄
            );
            // * 第三次同步，0号进程广播，并等待其他所有进程接收到主元
            MPI_Barrier(MPI_COMM_WORLD);
        } else {
            // * 这里表示其他进程向0号进程发送代表元素
            MPI_Send(samplePoints, sizeof(int) * split, MPI_CHAR, 0, ALLTOONE_TYPE + MyID, MPI_COMM_WORLD);
            // * 等待0号进程接收到全部数据
            MPI_Barrier(MPI_COMM_WORLD);
            // * 第二次同步，所有进程等待0号进程排序完成
            MPI_Barrier(MPI_COMM_WORLD);
            // * 这里其他进程接受0号进程发送的数据
            MPI_Bcast(samplePoints, sizeof(int) * (1 + split), MPI_CHAR, 0, MPI_COMM_WORLD);
            // * 第三次同步，当所有进程接受数据完成
            MPI_Barrier(MPI_COMM_WORLD);
        }

        /*
        * 每个处理器根据上步送来的P-1个主元把自己的n/P个数据分成P段，
            记为处理器Pi的第j+1段，其中i=0,…,P-1，j=0,…,P-1，
            对应于算法13.5步骤（5）
        */
        n = mylength;
        int current = 1;
        // * index[0] 表示第0段的开始位置
        index[0] = 0;
        /*
        * 找到第一个大于第一个元素的主元，之前的主元均小于等于第一个元素
            表示前面的分段均开始于0，结束于0号位置，因此将这一段的开始
            和结束位置设置为0
        */
        while ((data[0] >= samplePoints[current]) && (current < SumID)) {
            index[2 * current - 1] = 0;
            index[2 * current] = 0;
            current++;
        }
        // * 如果所有元素均大于最大的主元，则所有元素都在最后一段
        if (current == SumID) {
            index[2 * current - 1] = n;
        }
        /*
        * 源程序中使用c1, c2, c3, c4命名程序，十分不利于阅读
            这里进行重命名
            nameBefore      nameAfter   Meaning
            c1              left        表示二分查找中的左端点
            c2              mid         表示二分查找中的中间点mid
            c3              right       表示二分查找中的右端点
            c4              pivot       表示主元
        * 这里源程序主要进行的是二分查找，用主元在排好序的数组中找到对应的位置
            通过找到的结果对数组进行分段
        * 这里同时对源程序二分进行改进，保证:
            data[left] <= pivot
            data[right] > pivot
            并在此基础上不断迭代缩小left和right之间的差距，
            直到right = left + 1，此时right即为所求
        */
        int left, mid, right, pivot;
        for (left = 0; current < SumID; current++) {
            right = mylength;
            mid = (left + right) / 2;
            pivot = samplePoints[current];
            while (data[mid] != pivot && left < right) {
                if (data[mid] > pivot) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
                mid = (left + right) / 2;
            }
            while (data[mid] <= pivot && mid < mylength) {
                mid++;
            }
            if (mid == mylength) {
                /*
                * 如果mid等于最终的长度，说明当前段为最后一段，更新后续段的
                    开始和结束位置，最后更新current=total跳出循环
                */
                index[2 * current - 1] = mylength;
                for (i = current; i < SumID; i++) {
                    index[2 * i] = index[2 * i + 1] = mylength;
                }
                current = SumID;    // * 跳出循环
            } else {
                /*
                * 当前段结束位置为mid，下一段开始位置为mid
                */
                index[2 * current] = mid;
                index[2 * current - 1] = mid;
            }
            left = mid;
        }
        /*
        left = 0;
        while (current < SumID) {
            pivot = samplePoints[current];
            right = n;
            mid = (int)((left + right) / 2);
            // biSearch
            while ((data[mid] != pivot) && (left < right)) {
                if (data[mid] > pivot) {
                    right = mid - 1;
                    mid = (int)((left + right) / 2);
                } else {
                    left = mid + 1;
                    mid = (int)((left + right) / 2);
                }
            }
            while ((data[mid] <= pivot) && (mid < n))
                mid++;
            if (mid == n) {
                index[2 * current - 1] = n;
                for (k = current; k < SumID; k++) {
                    index[2 * k] = 0;
                    index[2 * k + 1] = 0;
                }
                current = SumID;
            } else {
                // 把主元i分割的位置记录一下
                index[2 * current] = mid;
                index[2 * current - 1] = mid;
            }
            left = mid;
            mid = (int)((left + right) / 2);
            current++;
        }*/
        if (current == SumID)
            index[2 * current - 1] = n;
        // * 进行一次同步，表示所有进程完成分段
        MPI_Barrier(MPI_COMM_WORLD);

        /*
        * 每个处理器送它的第i+1段给处理器Pi，
            从而使得第i个处理器含有所有处理器的第i段数据(i=0,…,P-1)，
            对应于算法13.5步骤（6）
        */
        int* segmentLength = samplePoints;
        j = 0;
        for (i = 0; i < SumID; i++) {
            if (i == MyID) {
                /*
                * i == MyID，表明这是需要发送的进程
                    这里segmentLength数组使用原来的samplePoints数组的位置
                    对于进程i，其中的segmentLength[j]表示进程j的第i段长度
                    本进程的可以直接计算得到
                */
                segmentLength[i] = index[2 * i + 1] - index[2 * i];
                for (n = 0; n < SumID; n++)
                    if (n != MyID) {
                        /* 向第 k 个处理器发送本机第 k + 1 段的长度 */
                        k = index[2 * n + 1] - index[2 * n];
                        MPI_Send(&k, sizeof(int), MPI_CHAR, n, MULTI_LEN + MyID, MPI_COMM_WORLD);
                    }
            } else {
                /*
                * 从第 i 个处理器获取其第 i + 1 段的长度，
                    存入 segmentLength[i]
                */
                MPI_Recv(&segmentLength[i], sizeof(int), MPI_CHAR, i, MULTI_LEN + i, MPI_COMM_WORLD, &status[j++]);
            }
        }
        // * 这里进行一次同步，表示所有长度信息发送/接收完毕
        MPI_Barrier(MPI_COMM_WORLD);
        /*
        * 接下来每个进程将向其他进程发送对应段的数据
            并从其他进程中得到对应段的数据
        * 第 i 个处理器向第 j 个处理器发送第 j 段
        * currentLength: 当前进程接收的数据的数量
        * l: 表示某一段的长度
        l ->
        */
        j = 0;
        l = 0;
        int currentLength = 0;
        for (i = 0; i < SumID; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == MyID) {
                // * 当前进程不用给自己发送数据，直接复制即可
                for (n = index[2 * i]; n < index[2 * i + 1]; n++)
                    buffer[currentLength++] = data[n];
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == MyID) {
                for (n = 0; n < SumID; n++) {
                    /*
                    * 向其他进程发送对应数据段
                    */
                    if (n != MyID) {
                        MPI_Send(&data[index[2 * n]], sizeof(int) * (index[2 * n + 1] - index[2 * n]), MPI_CHAR, n, MULTI_TYPE + MyID, MPI_COMM_WORLD);
                    }
                }
            } else {
                /*
                * 获得其他进程的第i段长度
                * 接收来自其他进程的第i段，维护length
                */
                l = segmentLength[i];
                MPI_Recv(&buffer[currentLength], l * sizeof(int), MPI_CHAR, i, MULTI_TYPE + i, MPI_COMM_WORLD, &status[j++]);
                currentLength = currentLength + l;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        mylength = currentLength;
        MPI_Barrier(MPI_COMM_WORLD);
        /*
        * 每个处理器再通过P路归并排序将上一步的到的数据排序；
            从而这n个数据便是有序的，对应于算法13.5步骤（7）
        */
        currentLength = 0;
        // multimerge(buffer, samplePoints, data, &currentLength, SumID);
        multimerge(buffer, data, samplePoints, SumID);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /*
    * 这里也有输出问题，按照同样的方法进行ring输出
    * 使用s一次性将结果输出
    */
    if (MyID != 0) {
        MPI_Recv(&mutex, 1, MPI_CHAR, MyID - 1, ONETOALL_TYPE, MPI_COMM_WORLD, status);
    }
    char* s = (char*)malloc(16 * DataSize);
    sprintf(s, "# On node %d the sorted data is : \n", MyID);
    for (i = 0; i < mylength; i++) {
        if (i + 1 != mylength) sprintf(s, "%s %d ", s, data[i]);
        else sprintf(s, "%s %d\n", s, data[i]);
    }
    fputs(s, stdout), free(s);
    if (MyID != SumID - 1) {
        MPI_Send(&mutex, 1, MPI_CHAR, MyID + 1, ONETOALL_TYPE, MPI_COMM_WORLD);
    }
}

/*输出错误信息*/
void merror(char *ch) {
    printf("%s\n", ch);
    exit(1);
}


/* 交换两数位置 */
void sswap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

/*
* 使用三数取中优化快排：选取数组开头，中间和结尾的元素，
    通过比较，选择中间的值作为快排的基准
*/
int NumberOfThree(int arr[], int low, int high) {
    int mid = low + ((high - low) >> 1);
    if (arr[mid] > arr[high]) {
		sswap(&arr[mid], &arr[high]);
	}
	if (arr[low] > arr[high]) {
		sswap(&arr[low], &arr[high]);
	}
	if (arr[mid] > arr[low]) {
		sswap(&arr[mid], &arr[low]);
	}
	//此时，arr[mid] <= arr[low] <= arr[high]
	return arr[low];
}

/*
* 这个函数确定划分范围
*/
int Partition(int a[], int p, int r) {
	int i = p, j = r+1;
	int x = NumberOfThree(a, p, r);
	while(1) {
		while(a[++i] < x && i < r);
		while(a[--j] > x);
		if(i >= j) break;
		sswap(&a[i], &a[j]);
	}
	a[p] = a[j];
	a[j] = x;
	return j;
}

void InsertSort(int arr[], int m, int n) {
	int i, j;
	int temp; // 用来存放临时的变量
	for(i = m+1; i <= n; i++)
	{
		temp = arr[i];
		for(j = i-1; (j >= m)&&(arr[j] > temp); j--)
		{
			arr[j + 1] = arr[j];
		}
		arr[j + 1] = temp;
	}
}

void qqqsot(int arr[],int low,int high) {
    int pivotPos;
    if (high - low + 1 < 10)
    {
        InsertSort(arr,low,high);
        return;
    }
    if(low < high)
    {
        pivotPos = Partition(arr,low,high);
        qqqsot(arr,low,pivotPos-1);
        qqqsot(arr,pivotPos+1,high);
    }
}


/*
* 当待排序列的长度达到一定数值后，再使用快排的优化并不大，可以使用插入排序
*/
/*串行快速排序算法*/
void quicksort(int *datas, int bb, int ee) {
    int tt, i, j;
    tt = datas[bb];
    i = bb;
    j = ee;

    if (i < j) {
        while (i < j) {
            while ((i < j) && (tt <= datas[j]))
                j--;
            if (i < j) {
                datas[i] = datas[j];
                i++;
                while ((i < j) && (tt > datas[i]))
                    i++;
                if (i < j) {
                    datas[j] = datas[i];
                    j--;
                    if (i == j)
                        datas[i] = tt;
                } else
                    datas[j] = tt;
            } else
                datas[i] = tt;
        }

        quicksort(datas, bb, i - 1);
        quicksort(datas, i + 1, ee);
    }
}

/*串行多路归并算法*/
/*void multimerge(int *data1, int *ind, int *data, int *iter, int SumID) {
    int i, j, n;

    j = 0;
    for (i = 0; i < SumID; i++)
        if (ind[i] > 0)
        {
            ind[j++] = ind[i];
            if (j < i + 1)
                ind[i] = 0;
        }
    if (j > 1)
    {
        n = 0;
        for (i = 0; i < j, i + 1 < j; i = i + 2)
        {
            merge(&(data1[n]), ind[i], ind[i + 1], &(data[n]));
            ind[i] += ind[i + 1];
            ind[i + 1] = 0;
            n += ind[i];
        }
        if (j % 2 == 1)
            for (i = 0; i < ind[j - 1]; i++)
                data[n++] = data1[n];
        (*iter)++;
        multimerge(data, ind, data1, iter, SumID);
    }
}

// * 合并两个有序数组
void merge(int *data1, int s1, int s2, int *data2) {
    int i, l, m;

    l = 0;
    m = s1;
    for (i = 0; i < s1 + s2; i++)
    {
        if (l == s1)
            data2[i] = data1[m++];
        else if (m == s2 + s1)
            data2[i] = data1[l++];
        else if (data1[l] > data1[m])
            data2[i] = data1[m++];
        else
            data2[i] = data1[l++];
    }
}*/

/*
* 这个函数的作用是合并两个有序数组
*/
void merge(const int src[], int dst[], int l, int mid, int r) {
    /*
    * 对于原输入格式进行修改，在原来的程序中s1表示输入数据的中间部分
        s1+s2表示所有的数据大小。
        输入数据中data2表示排序的目标位置，这里使用dst代替。
        输入数据中data1表述两个有序数组的存放位置，这里使用src代替
        同时添加l，表示左边边界。这样不必默认初始位置是src的0号位置，
        便于merge在其他地方使用
    */
    int l_loop = l;     // * 表示循环中左边数组的边界
    int r_loop = mid;   // * 表示循环中右边数组的边界
    int index = 0;      // * 表示dst中位置
    while (l_loop < mid && r_loop < r) {
        if (src[l_loop] < src[r_loop]) {
            dst[index++] = src[l_loop++];
        } else {
            dst[index++] = src[r_loop++];
        }
    }
    // * 如果右边数组先结束，则将左边数组剩下的值放到dst中
    while (l_loop < mid) {
        dst[index++] = src[l_loop++];
    }
    // * 如果左边数组先结束，则将右边数组剩下的值放到dst中
    while (r_loop < r) {
        dst[index++] = src[r_loop++];
    }
}

/*
* 这个函数将不断将左右两边递归排序，再将排好序的左右两边合并。
*/
void subArrayMerge(int src[], int buf[], int index[], int l, int r) {
    if (r - l < 1) return;
    int mid = (l + r) / 2;
    /*
     * 分别递归两边，得到两个有序子数组
     * 再合并得到的两个有序子数组
     */
    subArrayMerge(src, buf, index, l, mid);
    subArrayMerge(src, buf, index, mid + 1, r);
    merge(src, buf, index[l], index[mid + 1], index[r + 1]);
    /*
    * 原来的实现中，使用奇偶性判断最终结果在哪个数组中，
        这里固定使用src和buf分别表示最终结果和中间结果。
        因此当左右两边合并之后，需要位置最终结果在src中
    */
    int i;
    for (i = index[l]; i < index[r + 1]; i++) {
        src[i] = buf[i - index[l]];
    }
}

void multimerge(int src[], int buf[], const int length[], int total) {
    int* index = (int*)malloc((total + 1) * sizeof(int));
    index[0] = 0;
    /*
     * 输入的 length 意义是每一段子数组的长度
     * 为了在递归部分中快速得到每一段子数组的下标信息，需对 length 数组求和
     * 本质上是前缀和优化，实现 O(1) 得到下标
     */
    int i;
    for (i = 1; i <= total; i++) {
        index[i] = index[i - 1] + length[i - 1];
    }
    /*
     * 原实现本质上还是分治实现，不过其尾递归实现更像是“披着递归皮的循环”
     * 且其中存在较多特殊处理，如对 length 数组压缩和清零，对区间数的奇偶特判等
     * 考虑到其原实现不够清晰优雅，对此部分进行了重写
     */
    subArrayMerge(src, buf, index, 0, total - 1);
    free(index);
}