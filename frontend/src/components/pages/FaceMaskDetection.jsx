import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";

const FaceMaskDetection = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <h1>หน้าแสดงการทดสอบตรวจจับใบหน้า</h1>
      </PageContent>
    </AppContainer>
  );
};

export default FaceMaskDetection;
