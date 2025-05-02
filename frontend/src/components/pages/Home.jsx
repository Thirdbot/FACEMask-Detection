import React from "react";
import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";

const Home = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <h3>หน้าหลัก</h3>
      </PageContent>
    </AppContainer>
  );
};

export default Home;
