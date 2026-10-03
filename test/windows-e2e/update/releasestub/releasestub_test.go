package releasestub

import (
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func get(t *testing.T, c *http.Client, url string) (*http.Response, error) {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	return c.Do(req)
}

func client(s *Server) *http.Client {
	return &http.Client{Transport: s.Transport(), Timeout: 20 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
}

func TestEveryAuthorityURLIsServedLocallyAndCounted(t *testing.T) {
	s := New(t)
	exe := append([]byte("MZ"), make([]byte, 100)...)
	s.Publish("0.2.2-dev", exe)
	c := client(s)

	resp, err := get(t, c, "https://api.github.com/repos/yohn-jp/hachidori/releases?per_page=100&page=1")
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("list: %v %v", resp, err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(body), `"tag_name":"0.2.2-dev"`) || !strings.Contains(string(body), "hachidori-windows-amd64.exe.sha256") {
		t.Fatalf("list body %s", body)
	}
	resp, err = get(t, c, "https://github.com/yohn-jp/hachidori/releases/download/0.2.2-dev/hachidori-windows-amd64.exe")
	if err != nil || resp.StatusCode != http.StatusFound || resp.Header.Get("Location") != "https://objects.githubusercontent.com/blob/0.2.2-dev/hachidori-windows-amd64.exe" {
		t.Fatalf("download redirect: %v %v", resp, err)
	}
	resp.Body.Close()
	resp, err = get(t, c, resp.Header.Get("Location"))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(got) != string(exe) {
		t.Fatal("the asset body differs from the published executable")
	}
	if s.Count() != 3 || s.Strings()[0] != "GET api.github.com/repos/yohn-jp/hachidori/releases?per_page=100&page=1" {
		t.Fatalf("requests %v", s.Strings())
	}
}

func TestAHostOutsideTheAuthorityIsRefusedAndStillCannotLeaveTheMachine(t *testing.T) {
	s := New(t)
	resp, err := get(t, client(s), "https://example.org/anything")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusMisdirectedRequest {
		t.Fatalf("status %d", resp.StatusCode)
	}
	// The transport dialed the local listener whatever the URL said, which is
	// the point: the request was recorded here, not sent to example.org.
	if s.Count() != 1 || s.Requests()[0].Host != "example.org" {
		t.Fatalf("requests %v", s.Strings())
	}
}

func TestCredentialsAreRefused(t *testing.T) {
	s := New(t)
	req, _ := http.NewRequest(http.MethodGet, "https://api.github.com/repos/yohn-jp/hachidori/releases", nil)
	req.Header.Set("Authorization", "Bearer x")
	resp, err := client(s).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("a request with credentials was served: %d", resp.StatusCode)
	}
}

func TestHeldDownloadPausesThenCompletesOrAborts(t *testing.T) {
	s := New(t)
	exe := make([]byte, 4096)
	s.Publish("0.2.2-dev", exe)
	url := "https://" + AssetHost + "/blob/0.2.2-dev/hachidori-windows-amd64.exe"

	g := s.HoldExe("0.2.2-dev", 1000)
	resp, err := get(t, client(s), url)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-g.Reached():
	case <-time.After(10 * time.Second):
		t.Fatal("the gate was never reached")
	}
	first := make([]byte, 1000)
	if _, err := io.ReadFull(resp.Body, first); err != nil {
		t.Fatalf("the bytes before the gate must arrive: %v", err)
	}
	g.Release()
	rest, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil || len(rest) != 3096 {
		t.Fatalf("after release: %d bytes, %v", len(rest), err)
	}

	g = s.HoldExe("0.2.2-dev", 100)
	resp, err = get(t, client(s), url)
	if err != nil {
		t.Fatal(err)
	}
	<-g.Reached()
	g.Abort()
	if _, err := io.ReadAll(resp.Body); err == nil {
		t.Fatal("an aborted transfer must not end cleanly")
	}
	resp.Body.Close()
}

func TestFaultsApplyUntilHealed(t *testing.T) {
	s := New(t)
	s.Publish("0.2.2-dev", make([]byte, 2048))
	list := "https://" + APIHost + "/repos/yohn-jp/hachidori/releases?per_page=100&page=1"
	c := client(s)

	s.FailList(503)
	if resp, err := get(t, c, list); err != nil || resp.StatusCode != 503 {
		t.Fatalf("%v %v", resp, err)
	}
	s.Heal()
	s.DropList()
	if resp, err := get(t, c, list); err == nil {
		resp.Body.Close()
		t.Fatal("a dropped list answered")
	}
	s.Heal()
	if resp, err := get(t, c, list); err != nil || resp.StatusCode != 200 {
		t.Fatalf("after heal: %v %v", resp, err)
	}

	s.CutExe("0.2.2-dev", 500)
	resp, err := get(t, c, "https://"+AssetHost+"/blob/0.2.2-dev/hachidori-windows-amd64.exe")
	if err != nil {
		t.Fatal(err)
	}
	if b, err := io.ReadAll(resp.Body); err == nil || len(b) > 500 {
		t.Fatalf("a cut transfer: %d bytes, %v", len(b), err)
	}
	resp.Body.Close()
}

func TestHeldListWaitsForTheTest(t *testing.T) {
	s := New(t)
	g := s.HoldList()
	done := make(chan int, 1)
	go func() {
		resp, err := get(t, client(s), "https://"+APIHost+"/repos/yohn-jp/hachidori/releases?per_page=100&page=1")
		if err != nil {
			done <- -1
			return
		}
		resp.Body.Close()
		done <- resp.StatusCode
	}()
	<-g.Reached()
	select {
	case <-done:
		t.Fatal("the held list answered before release")
	default:
	}
	g.Release()
	if code := <-done; code != 200 {
		t.Fatalf("status %d", code)
	}
}
